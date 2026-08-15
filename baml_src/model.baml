enum ModelVariant {
    Fp16,
    Int8,
}

class ModelAsset {
    remote_name: string,
    local_name: string,
    size: int,
    sha256: string,
}

function model_assets(variant: ModelVariant) -> ModelAsset[] {
    let vocabulary = ModelAsset {
        remote_name: "vocab.txt",
        local_name: "vocab.txt",
        size: 102132,
        sha256: "ba8e4007c65f4bb4358ffe2ecc13d9ccc7a10351151065242b5c3a943e685742",
    };
    match (variant) {
        ModelVariant.Fp16 => {
            [
                ModelAsset {
                    remote_name: "encoder-model.fp16.onnx",
                    local_name: "encoder-model.onnx",
                    size: 1238960452,
                    sha256: "a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717",
                },
                ModelAsset {
                    remote_name: "decoder_joint-model.fp16.onnx",
                    local_name: "decoder_joint-model.onnx",
                    size: 36266140,
                    sha256: "b33a73b7c1d71b9d5a0911f5cb478be3dcbf79f53355c531ab1cd1dcd68ad8ef",
                },
                vocabulary,
            ]
        },
        ModelVariant.Int8 => {
            [
                ModelAsset {
                    remote_name: "encoder-model.int8.onnx",
                    local_name: "encoder-model.onnx",
                    size: 652183999,
                    sha256: "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
                },
                ModelAsset {
                    remote_name: "decoder_joint-model.int8.onnx",
                    local_name: "decoder_joint-model.onnx",
                    size: 18202004,
                    sha256: "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
                },
                vocabulary,
            ]
        },
    }
}

function model_variant_name(variant: ModelVariant) -> string {
    match (variant) {
        ModelVariant.Fp16 => "FP16",
        ModelVariant.Int8 => "INT8",
    }
}

function model_variant_cache_dir(variant: ModelVariant) -> string {
    match (variant) {
        ModelVariant.Fp16 => "parakeet-tdt-0.6b-v3-fp16-f88260fa",
        ModelVariant.Int8 => "parakeet-tdt-0.6b-v3-int8-f88260fa",
    }
}

function cache_root() -> string {
    let root = if let xdg: string = baml.env.get("XDG_CACHE_HOME") {
        xdg
    } else if let home: string = baml.env.get("HOME") {
        join_path(home, ".cache")
    } else {
        invalid_argument("HOME or XDG_CACHE_HOME is required")
    };
    join_path(root, "local-wisper")
}

function asset_has_expected_size(model_dir: string, asset: ModelAsset) -> bool {
    let path = join_path(model_dir, asset.local_name);
    baml.fs.exists(path) && baml.fs.size(path) == asset.size
}

function verify_sha256(path: string, expected: string) -> bool {
    let output = baml.sys.exec("sha256sum", ["--", path], null);
    if (!output.ok()) {
        return false;
    }
    let digest = baml.String.from_utf8(output.stdout).trim().split(" ")[0];
    digest == expected
}

function download_model_asset(model_dir: string, asset: ModelAsset) -> null {
    let repository = "ysdede/parakeet-tdt-0.6b-v3-onnx";
    let revision = "f88260fa0777fe0868dda6df85d1a98f012a4a7a";
    let destination = join_path(model_dir, asset.local_name);
    let part = `${destination}.part`;
    let url = `https://huggingface.co/${repository}/resolve/${revision}/${asset.remote_name}`;
    baml.io.eprintln(`downloading ${asset.remote_name}`);
    let download = baml.sys.exec(
        "curl",
        ["--fail", "--location", "--continue-at", "-", "--output", part, url],
        null,
    );
    if (!download.ok()) {
        invalid_argument(`failed to download ${asset.remote_name}`)
    }
    let valid = baml.fs.size(part) == asset.size && verify_sha256(part, asset.sha256);
    if (!valid) {
        remove_file_if_present(part);
        invalid_argument(`downloaded ${asset.remote_name} failed size or SHA-256 verification`)
    }
    let moved = baml.sys.exec("mv", ["--", part, destination], null);
    if (!moved.ok()) {
        invalid_argument(`failed to commit ${asset.local_name}`)
    }
    null
}

function prepare_model_variant(variant: ModelVariant) -> string {
    let model_dir = join_path(join_path(cache_root(), "models"), model_variant_cache_dir(variant));
    baml.fs.mkdir(model_dir, baml.fs.MkdirOptions { recursive: true });
    let marker = join_path(model_dir, ".complete");
    let assets = model_assets(variant);
    if (
        baml.fs.exists(marker)
            && assets.every((asset) -> {
                asset_has_expected_size(model_dir, asset)
            })
    ) {
        return model_dir;
    }
    for (let asset in assets) {
        let destination = join_path(model_dir, asset.local_name);
        if (asset_has_expected_size(model_dir, asset) && verify_sha256(destination, asset.sha256)) {
            continue;
        }
        download_model_asset(model_dir, asset);
    }
    let repository = "ysdede/parakeet-tdt-0.6b-v3-onnx";
    let revision = "f88260fa0777fe0868dda6df85d1a98f012a4a7a";
    atomic_write(marker, `${repository}@${revision} ${model_variant_name(variant)}\n`);
    model_dir
}

function cuda_hardware_present() -> bool {
    let probe = baml.sys.exec("nvidia-smi", ["-L"], null) catch_all (error) {
        _ => {
            return false;
        },
    };
    probe.ok()
}

function load_cuda_model(
    native_load_model: (
        model_dir: string,
        variant: ModelVariant,
    ) -> null throws baml.errors.HostCallable,
) -> null {
    native_load_model(prepare_model_variant(ModelVariant.Fp16), ModelVariant.Fp16)
}

function try_load_cuda_model(
    native_load_model: (
        model_dir: string,
        variant: ModelVariant,
    ) -> null throws baml.errors.HostCallable,
) -> bool {
    load_cuda_model(native_load_model) catch_all (error) {
        _ => {
            baml.io.eprintln(
                `CUDA model initialization failed; falling back to CPU: ${error.to_string()}`,
            );
            return false;
        },
    };
    true
}

function initialize_model(
    preference: DevicePreference,
    native_load_model: (
        model_dir: string,
        variant: ModelVariant,
    ) -> null throws baml.errors.HostCallable,
) -> null {
    if (preference == DevicePreference.Cpu) {
        return native_load_model(prepare_model_variant(ModelVariant.Int8), ModelVariant.Int8);
    }
    if (cuda_hardware_present()) {
        if (try_load_cuda_model(native_load_model)) {
            return null;
        }
    } else {
        baml.io.eprintln("no NVIDIA CUDA device detected; using CPU");
    }
    native_load_model(prepare_model_variant(ModelVariant.Int8), ModelVariant.Int8)
}

test "model variants use the filenames expected by Parakeet" {
    for (let variant in [ModelVariant.Fp16, ModelVariant.Int8]) {
        assert.equal(
            model_assets(variant).map((asset) -> {
                asset.local_name
            }),
            ["encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt"],
        )
    }
}
