use std::fs::{self, OpenOptions};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OpenFlags, TransactionBehavior, params};

use crate::paths;

const DATABASE_NAME: &str = "transcripts.sqlite3";
const SCHEMA_VERSION: i64 = 1;
const CREATE_SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS transcripts (
        id INTEGER PRIMARY KEY,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
        raw_text TEXT NOT NULL,
        final_text TEXT NOT NULL,
        post_process_model TEXT,
        processing TEXT NOT NULL
            CHECK (processing IN ('local', 'model', 'model_timeout_fallback', 'model_error_fallback', 'model_rejected_fallback'))
    );
";

pub fn save(transcript: baml_sdk::TranscriptRecord) -> Result<()> {
    let database_path = paths::data_dir()?.join(DATABASE_NAME);
    save_to(&database_path, &transcript)
}

fn save_to(database_path: &Path, transcript: &baml_sdk::TranscriptRecord) -> Result<()> {
    create_secure_database_file(database_path)?;
    let mut connection = Connection::open_with_flags(
        database_path,
        OpenFlags::default() | OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )
    .with_context(|| {
        format!(
            "failed to open transcript database {}",
            database_path.display()
        )
    })?;
    configure_connection(&connection)?;
    initialize_schema(&mut connection)?;
    connection
        .execute(
            "INSERT INTO transcripts (raw_text, final_text, post_process_model, processing) VALUES (?1, ?2, ?3, ?4)",
            params![
                &transcript.raw_text,
                &transcript.final_text,
                &transcript.post_process_model,
                processing_name(transcript.processing),
            ],
        )
        .context("failed to save transcript")?;
    Ok(())
}

fn configure_connection(connection: &Connection) -> Result<()> {
    connection
        .busy_timeout(Duration::from_secs(1))
        .context("failed to configure transcript database timeout")?;
    let journal_mode = connection
        .query_row("PRAGMA journal_mode = WAL", [], |row| {
            row.get::<_, String>(0)
        })
        .context("failed to enable transcript database WAL mode")?;
    if !journal_mode.eq_ignore_ascii_case("wal") {
        bail!("transcript database refused WAL mode and selected {journal_mode}")
    }
    Ok(())
}

fn initialize_schema(connection: &mut Connection) -> Result<()> {
    if schema_version(connection)? == SCHEMA_VERSION {
        return Ok(());
    }
    let transaction = connection
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .context("failed to lock transcript database for schema initialization")?;
    // Another process may have migrated while this one waited for the lock.
    let version = schema_version(&transaction)?;
    match version {
        0 if database_is_empty(&transaction)? => transaction
            .execute_batch(CREATE_SCHEMA)
            .context("failed to initialize transcript database")?,
        0 => bail!("refusing to initialize a non-empty unversioned transcript database"),
        SCHEMA_VERSION => {}
        other if other > SCHEMA_VERSION => bail!(
            "transcript database schema version {version} is newer than supported version {SCHEMA_VERSION}"
        ),
        _ => bail!("unsupported transcript database schema version {version}"),
    }
    if version != SCHEMA_VERSION {
        transaction
            .pragma_update(None, "user_version", SCHEMA_VERSION)
            .context("failed to update transcript database schema version")?;
    }
    transaction
        .commit()
        .context("failed to commit transcript database schema initialization")
}

fn schema_version(connection: &Connection) -> Result<i64> {
    connection
        .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
        .context("failed to read transcript database schema version")
}

fn database_is_empty(connection: &Connection) -> Result<bool> {
    connection
        .query_row(
            "SELECT NOT EXISTS (SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%')",
            [],
            |row| row.get(0),
        )
        .context("failed to inspect unversioned transcript database")
}

fn processing_name(processing: baml_sdk::TranscriptProcessing) -> &'static str {
    match processing {
        baml_sdk::TranscriptProcessing::Local => "local",
        baml_sdk::TranscriptProcessing::Model => "model",
        baml_sdk::TranscriptProcessing::ModelTimeoutFallback => "model_timeout_fallback",
        baml_sdk::TranscriptProcessing::ModelErrorFallback => "model_error_fallback",
        baml_sdk::TranscriptProcessing::ModelRejectedFallback => "model_rejected_fallback",
    }
}

fn create_secure_database_file(path: &Path) -> Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("failed to create transcript database {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect transcript database {}", path.display()))?;
    let uid = unsafe { libc::geteuid() };
    if !metadata.is_file() || metadata.uid() != uid {
        bail!(
            "transcript database {} is not a regular file owned by user {uid}",
            path.display()
        )
    }
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .with_context(|| format!("failed to secure transcript database {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn transcript(
        raw_text: &str,
        final_text: &str,
        processing: baml_sdk::TranscriptProcessing,
        post_process_model: Option<&str>,
    ) -> baml_sdk::TranscriptRecord {
        baml_sdk::TranscriptRecord {
            raw_text: raw_text.to_owned(),
            final_text: final_text.to_owned(),
            processing,
            post_process_model: post_process_model.map(str::to_owned),
        }
    }

    fn all_processing_outcomes() -> [baml_sdk::TranscriptProcessing; 5] {
        [
            baml_sdk::TranscriptProcessing::Local,
            baml_sdk::TranscriptProcessing::Model,
            baml_sdk::TranscriptProcessing::ModelTimeoutFallback,
            baml_sdk::TranscriptProcessing::ModelErrorFallback,
            baml_sdk::TranscriptProcessing::ModelRejectedFallback,
        ]
    }

    #[test]
    fn saves_raw_and_final_transcripts_in_a_private_database() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "local-wisper-history-test-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let database_path = directory.join(DATABASE_NAME);

        for processing in all_processing_outcomes() {
            save_to(
                &database_path,
                &transcript(
                    "raw words",
                    "Final words.",
                    processing,
                    Some("gpt-5.6-luna"),
                ),
            )
            .unwrap();
        }

        let connection = Connection::open(&database_path).unwrap();
        let saved = connection
            .query_row(
                "SELECT raw_text, final_text, post_process_model, processing FROM transcripts WHERE id = 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            saved,
            (
                "raw words".to_owned(),
                "Final words.".to_owned(),
                Some("gpt-5.6-luna".to_owned()),
                "local".to_owned(),
            )
        );
        let outcomes = {
            let mut statement = connection
                .prepare("SELECT processing FROM transcripts ORDER BY id")
                .unwrap();
            statement
                .query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
        };
        assert_eq!(
            outcomes,
            [
                "local",
                "model",
                "model_timeout_fallback",
                "model_error_fallback",
                "model_rejected_fallback",
            ]
        );
        assert_eq!(
            connection
                .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
                .unwrap(),
            SCHEMA_VERSION
        );
        assert_eq!(
            connection
                .query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
                .unwrap(),
            "wal"
        );
        assert_eq!(
            fs::metadata(&database_path).unwrap().permissions().mode() & 0o777,
            0o600
        );

        drop(connection);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn refuses_a_symlinked_database() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "local-wisper-history-symlink-test-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let target = directory.join("target");
        fs::write(&target, "untouched").unwrap();
        let database_path = directory.join(DATABASE_NAME);
        symlink(&target, &database_path).unwrap();

        let error = save_to(
            &database_path,
            &transcript(
                "raw words",
                "Final words.",
                baml_sdk::TranscriptProcessing::Local,
                None,
            ),
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("failed to create transcript database")
        );
        assert_eq!(fs::read_to_string(target).unwrap(), "untouched");

        fs::remove_dir_all(directory).unwrap();
    }
}
