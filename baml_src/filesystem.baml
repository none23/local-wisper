function join_path(parent: string, child: string) -> string {
    if (parent.ends_with("/")) {
        `${parent}${child}`
    } else {
        `${parent}/${child}`
    }
}

function remove_file_if_present(path: string) -> null {
    if (baml.fs.exists(path)) {
        baml.fs.remove(path) catch_all (error) {
            _ => null
        }
    } else {
        null
    }
}

function atomic_write(path: string, contents: string) -> null {
    let part = `${path}.part-${baml.id.new()}`;
    let _ = baml.fs.write(part, contents);
    let moved = baml.sys.exec("mv", ["--", part, path], null);
    if (!moved.ok()) {
        remove_file_if_present(part);
        throw baml.errors.Io { message: `failed to commit ${path}` }
    }
    null
}
