#!/usr/bin/env bash
# Vendoring helper — run this ONCE locally before first commit.
# It clones the upstream zkllm-ccs2024 into ./zkllm/ and applies the
# same patches that the notebook applies at runtime (self-attn.cu,
# ffn.cu, Makefile), so that the Dockerfile can just `make` without
# any runtime git-clone + sed gymnastics.
#
# Usage:   bash scripts/vendor_zkllm.sh
# Result:  ./zkllm/  populated with patched source + Makefile

set -euxo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/zkllm"

if [[ -f "$DEST/Makefile" && -f "$DEST/self-attn.cu" ]]; then
    echo "zkllm/ already populated. Delete it first if you want to re-vendor."
    exit 0
fi

TMPDIR="$(mktemp -d)"
trap "rm -rf $TMPDIR" EXIT

git clone --depth 1 https://github.com/jvhs0706/zkllm-ccs2024.git "$TMPDIR/src"
# Move everything (including hidden files) to $DEST
mkdir -p "$DEST"
shopt -s dotglob
mv "$TMPDIR/src"/* "$DEST"/
shopt -u dotglob

cd "$DEST"

# --- Patch Makefile: use /usr/local/cuda instead of $CONDA_PREFIX ---
python3 - <<'PY'
import re
with open('Makefile') as f:
    mf = f.read()
mf = mf.replace('NVCC := $(CONDA_PREFIX)/bin/nvcc', 'NVCC := /usr/local/cuda/bin/nvcc')
mf = mf.replace('INCLUDES := -I$(CONDA_PREFIX)/include', 'INCLUDES := -I/usr/local/cuda/include')
mf = mf.replace('LIBS := -L$(CONDA_PREFIX)/lib', 'LIBS := -L/usr/local/cuda/lib64')
mf = re.sub(r'ARCH := sm_\d+', 'ARCH := sm_90', mf)
with open('Makefile', 'w') as f:
    f.write(mf)
print("Makefile patched (paths + ARCH=sm_90)")
PY

# --- Patch self-attn.cu: n_heads as CLI arg + zkSoftmax tuning ---
cat > self-attn.cu <<'EOF'
#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 8) {
        cerr << "Usage: " << argv[0]
             << " <linear|attn> <input> <seq_len> <embed_dim>"
                " <workdir> <layer_prefix> <o> [n_heads]" << endl;
        return 1;
    }
    string mode             = argv[1];
    string input_file_name  = argv[2];
    uint   seq_len          = std::stoi(argv[3]);
    uint   embed_dim        = std::stoi(argv[4]);
    string workdir          = argv[5];
    string layer_prefix     = argv[6];
    string output_file_name = argv[7];
    uint   n_heads          = (argc >= 9) ? std::stoi(argv[8]) : 32;

    if (mode == "linear") {
        auto X   = FrTensor::from_int_bin(input_file_name);
        auto Q_w = FrTensor::from_int_bin(workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin");
        auto K_w = FrTensor::from_int_bin(workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin");
        auto V_w = FrTensor::from_int_bin(workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin");

        auto Q = FrTensor::matmul(X, Q_w, seq_len, embed_dim, embed_dim);
        auto K = FrTensor::matmul(X, K_w, seq_len, embed_dim, embed_dim);
        auto V = FrTensor::matmul(X, V_w, seq_len, embed_dim, embed_dim);

        Q.save_int("temp_Q.bin");
        K.save_int("temp_K.bin");
        V.save_int("temp_V.bin");
        cout << "Linear projection done. Q,K,V saved." << endl;
        return 0;
    }
    else if (mode == "attn") {
        auto Q_full = FrTensor::from_int_bin("temp_Q.bin");
        auto K_full = FrTensor::from_int_bin("temp_K.bin");
        auto V_full = FrTensor::from_int_bin("temp_V.bin");

        uint d         = embed_dim / n_heads;
        uint head_size = seq_len * d;
        FrTensor full_out(seq_len * embed_dim);

        for (uint h = 0; h < n_heads; h++) {
            FrTensor Q(head_size, Q_full.gpu_data + h * head_size);
            FrTensor K(head_size, K_full.gpu_data + h * head_size);
            FrTensor V(head_size, V_full.gpu_data + h * head_size);

            auto X_scores = FrTensor::matmul(Q, K.transpose(seq_len, d), seq_len, d, seq_len);

            zkSoftmax softmax({1<<8, 1<<14, 1<<14}, 1, 0, 1UL<<32, {1<<14, 1<<18}, seq_len, seq_len, d, 1);
            Rescaling rs1(1<<20), rs2(1<<20);

            FrTensor shift(seq_len), X_shifted(seq_len * seq_len);
            vector<FrTensor> X_segments, Y_segments, m_segments;
            FrTensor Y = softmax.compute(X_scores, shift, X_shifted, X_segments, Y_segments, m_segments);

            auto head_out = FrTensor::matmul(Y, V, seq_len, seq_len, d);
            auto head_out_scaled = rs1(rs2(head_out));

            cudaMemcpy(full_out.gpu_data + h * head_size,
                       head_out_scaled.gpu_data,
                       head_size * sizeof(Fr_t),
                       cudaMemcpyDeviceToDevice);
        }

        full_out.save_int(output_file_name);
        cout << "Attention proof complete." << endl;
        return 0;
    }

    cerr << "Unknown mode: " << mode << endl;
    return 1;
}
EOF
echo "self-attn.cu patched"

# --- Patch ffn.cu: swiglu table 2^22 -> 2^20 ---
python3 - <<'PY'
import re
with open('ffn.cu') as f:
    src = f.read()
new_src = re.sub(
    r"tLookupRangeMapping swiglu\(-\(1 << \d+\), 1 << \d+, swiglu_values\);",
    "tLookupRangeMapping swiglu(-(1 << 19), 1 << 20, swiglu_values);",
    src,
)
with open('ffn.cu', 'w') as f:
    f.write(new_src)
print("ffn.cu patched")
PY

echo "Vendoring complete."
