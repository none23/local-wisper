enum ModelVariant {
    Fp32,
    Int8,
}

class ModelAsset {
    remote_name: string,
    local_name: string,
    size: int,
    sha256: string,
}

function model_assets(variant: ModelVariant) -> ModelAsset[] {
    let tokenizer = ModelAsset {
        remote_name: "tokenizer.model",
        local_name: "tokenizer.model",
        size: 251056,
        sha256: "07d4e5a63840a53ab2d4d106d2874768143fb3fbdd47938b3910d2da05bfb0a9",
    };
    match (variant) {
        ModelVariant.Fp32 => {
            [
                ModelAsset {
                    remote_name: "encoder.onnx",
                    local_name: "encoder.onnx",
                    size: 41901475,
                    sha256: "50083dd0b2af503b87ea8fcf9fd7d7dc15dda3130e4113017c8b37ed00418ba4",
                },
                ModelAsset {
                    remote_name: "encoder.onnx.data",
                    local_name: "encoder.onnx.data",
                    size: 2437091328,
                    sha256: "c054b2932ee13aa39ed84982a399822f3193311fb6fd8e1e204566722614bd87",
                },
                ModelAsset {
                    remote_name: "decoder_joint.onnx",
                    local_name: "decoder_joint.onnx",
                    size: 35779240,
                    sha256: "64648c91935ea4819e9c31ea8a20f011d3b1c960be3861cb8bc74df0259cd998",
                },
                tokenizer,
            ]
        },
        ModelVariant.Int8 => {
            [
                ModelAsset {
                    remote_name: "encoder.int8.onnx",
                    local_name: "encoder.int8.onnx",
                    size: 42606669,
                    sha256: "c81adfab77634e00c1668a221a14f244c5fb3409e7c14eeebaf6ac963425910f",
                },
                ModelAsset {
                    remote_name: "encoder.int8.onnx.data",
                    local_name: "encoder.int8.onnx.data",
                    size: 611491584,
                    sha256: "3d54dd04646c15677bd2844a84df3770b12cc1ce183481f7b6e0def31c92114a",
                },
                ModelAsset {
                    remote_name: "decoder_joint.int8.onnx",
                    local_name: "decoder_joint.int8.onnx",
                    size: 8995064,
                    sha256: "7f76ad5f35035f25630075699c6c942a2c0c05ff42cb398f966f3c256d148e1e",
                },
                tokenizer,
            ]
        },
    }
}

function model_variant_name(variant: ModelVariant) -> string {
    match (variant) {
        ModelVariant.Fp32 => "FP32",
        ModelVariant.Int8 => "INT8",
    }
}

function model_variant_cache_dir(variant: ModelVariant) -> string {
    match (variant) {
        ModelVariant.Fp32 => "parakeet-unified-en-0.6b-fp32-09e90603",
        ModelVariant.Int8 => "parakeet-unified-en-0.6b-int8-09e90603",
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
    let repository = "bobNight/parakeet-unified-en-0.6b-onnx";
    let revision = "09e9060322d99c5f070010724786e6ee090fd51d";
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
    let repository = "bobNight/parakeet-unified-en-0.6b-onnx";
    let revision = "09e9060322d99c5f070010724786e6ee090fd51d";
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
    native_load_model(prepare_model_variant(ModelVariant.Fp32), ModelVariant.Fp32)
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

test "model variants use the filenames expected by Parakeet Unified" {
    assert.equal(
        model_assets(ModelVariant.Fp32).map((asset) -> {
            asset.local_name
        }),
        ["encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "tokenizer.model"],
    );
    assert.equal(
        model_assets(ModelVariant.Int8).map((asset) -> {
            asset.local_name
        }),
        [
            "encoder.int8.onnx",
            "encoder.int8.onnx.data",
            "decoder_joint.int8.onnx",
            "tokenizer.model",
        ],
    )
}
