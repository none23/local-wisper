use std::io::Write;
use std::process::{Command, Stdio};

pub fn type_text(text: &str) -> bool {
    Command::new("wtype")
        .arg(text)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

pub fn copy_text(text: &str) -> bool {
    [
        ("wl-copy", &[][..]),
        ("xclip", &["-selection", "clipboard"] as &[&str]),
        ("xsel", &["--clipboard", "--input"] as &[&str]),
    ]
    .iter()
    .any(|(program, args)| pipe_text(program, args, text))
}

fn pipe_text(program: &str, args: &[&str], text: &str) -> bool {
    let Ok(mut child) = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    else {
        return false;
    };
    let written = child
        .stdin
        .take()
        .map(|mut stdin| stdin.write_all(text.as_bytes()).is_ok())
        .unwrap_or(false);
    written && child.wait().map(|status| status.success()).unwrap_or(false)
}
