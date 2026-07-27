# ===============================================================
# EncryptX — Forensic Evidence Transmission System
# FILE: app.py
#
# APPROACH:
#   Encrypted pixel data is saved losslessly on the SERVER.
#   The user downloads a real JFIF image (just for viewing).
#   The JFIF filename contains a unique ID.
#   On decrypt, user uploads the JFIF — server reads the ID
#   from the filename, loads the lossless data, and decrypts.
#
#   This way:
#   - User gets a real viewable JFIF encrypted image
#   - Decryption is always perfect (uses lossless server data)
#   - No unsafe file format warnings
# ===============================================================

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pickle
import os
import math
import base64
import io
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

MOD        = 257
BLOCK_SIZE = 9
MU         = 1.0


# ── CRYPTO CORE ──────────────────────────────────────────────

def generate_T_matrix(mod=MOD):
    raw_bytes = os.urandom(8)
    seed      = int.from_bytes(raw_bytes, byteorder="big") % (2**31)
    rng       = np.random.RandomState(seed)
    flat      = rng.randint(0, mod, size=(BLOCK_SIZE * BLOCK_SIZE))
    T         = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
    idx       = 0
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if j >= i:
                T[i, j] = flat[idx] % mod
            idx += 1
    np.fill_diagonal(T, (np.diag(T) % (mod - 1)) + 1)
    return T, seed


def generate_chaotic_sequence(T_matrix, length, mod=MOD):
    trace_val = int(np.trace(T_matrix))
    x         = (trace_val % (mod - 1)) / mod
    y         = 1.0 - x
    chaos     = []
    for _ in range(length):
        x = abs(math.sin(math.pi * MU * (y + 3) * x * (1 - x)))
        y = abs(math.sin(math.pi * MU * (x + 3) * y * (1 - y)))
        chaos.append((x + y) / 2.0)
    return (np.array(chaos) * (mod - 1)).astype(int) % mod


def mod_matrix_inverse(T, mod=MOD):
    n   = T.shape[0]
    aug = np.hstack([T.astype(int) % mod, np.eye(n, dtype=int)])
    for col in range(n):
        pivot = int(aug[col, col]) % mod
        if pivot == 0:
            aug[col, col] = 1
            pivot         = 1
        p_inv    = pow(pivot, mod - 2, mod)
        aug[col] = (aug[col] * p_inv) % mod
        for row in range(n):
            if row != col:
                f        = int(aug[row, col]) % mod
                aug[row] = (aug[row] - f * aug[col]) % mod
    return aug[:, n:] % mod


def bijective_encrypt(data_vector, chaotic_int, T_matrix, mod=MOD):
    T_inv   = mod_matrix_inverse(T_matrix, mod)
    n       = len(data_vector)
    pad_len = (BLOCK_SIZE - n % BLOCK_SIZE) % BLOCK_SIZE
    a_pad   = np.pad(data_vector.astype(int), (0, pad_len))
    b_pad   = np.pad(chaotic_int,             (0, pad_len))
    output  = np.zeros(len(a_pad), dtype=int)
    for i in range(0, len(a_pad), BLOCK_SIZE):
        a_blk  = a_pad[i:i+BLOCK_SIZE]
        b_blk  = b_pad[i:i+BLOCK_SIZE]
        Ta     = (T_matrix @ a_blk) % mod
        Tb     = (T_matrix @ b_blk) % mod
        output[i:i+BLOCK_SIZE] = (T_inv @ (Ta + Tb)) % mod
    return output[:n].astype(np.uint8)


def bijective_decrypt(cipher_vector, chaotic_int, T_matrix, mod=MOD):
    T_inv   = mod_matrix_inverse(T_matrix, mod)
    n       = len(cipher_vector)
    pad_len = (BLOCK_SIZE - n % BLOCK_SIZE) % BLOCK_SIZE
    q_pad   = np.pad(cipher_vector.astype(int), (0, pad_len))
    b_pad   = np.pad(chaotic_int,               (0, pad_len))
    output  = np.zeros(len(q_pad), dtype=int)
    for i in range(0, len(q_pad), BLOCK_SIZE):
        q_blk  = q_pad[i:i+BLOCK_SIZE]
        b_blk  = b_pad[i:i+BLOCK_SIZE]
        TQ     = (T_matrix @ q_blk) % mod
        Tb     = (T_matrix @ b_blk) % mod
        output[i:i+BLOCK_SIZE] = (T_inv @ ((TQ - Tb) % mod)) % mod
    return output[:n].astype(np.uint8)


def secondary_diffusion(cipher_data, chaotic_int):
    seed        = int(chaotic_int[0]) * 1000 + int(chaotic_int[1])
    np.random.seed(seed % (2**31))
    raw_key     = np.random.randint(0, 2**31)
    add_key     = int(45 + (raw_key % 6))
    permutation = np.random.permutation(len(cipher_data))
    diffused    = ((cipher_data.astype(np.int32)[permutation] + add_key) % 256).astype(np.uint8)
    return diffused, permutation, add_key


def reverse_diffusion(cipher_data, permutation, add_key):
    unshifted             = ((cipher_data.astype(np.int32) - add_key) % 256).astype(np.uint8)
    inv_perm              = np.zeros_like(permutation)
    inv_perm[permutation] = np.arange(len(permutation))
    return unshifted[inv_perm]


def compute_npcr_uaci(original, encrypted):
    O    = original.astype(np.float64)
    E    = encrypted.astype(np.float64)
    npcr = (np.sum(O != E) / O.size) * 100.0
    uaci = (np.sum(np.abs(O - E) / 255.0) / O.size) * 100.0
    return round(npcr, 4), round(uaci, 4)


def array_to_b64(arr):
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/encrypt", methods=["POST"])
def encrypt():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img            = Image.open(file.stream).convert("RGB")
        arr            = np.array(img)
        original_shape = arr.shape
        data_vector    = arr.flatten().astype(np.uint8)

        T_matrix, seed  = generate_T_matrix()
        chaotic_int     = generate_chaotic_sequence(T_matrix, len(data_vector))
        cipher_data     = bijective_encrypt(data_vector, chaotic_int, T_matrix)
        final_cipher, permutation, add_key = secondary_diffusion(cipher_data, chaotic_int)
        npcr, uaci      = compute_npcr_uaci(data_vector, final_cipher)

        # Generate unique ID for this encryption session
        enc_id = hex(seed)[2:10].upper()

        # ── Save lossless data on server (for perfect decryption) ──
        lossless_path = os.path.join(app.config['OUTPUT_FOLDER'], f"data_{enc_id}.npz")
        np.savez_compressed(
            lossless_path,
            cipher         = final_cipher,
            original_shape = np.array(original_shape),
            T_matrix       = T_matrix,
            permutation    = permutation,
            add_key        = np.array([add_key])
        )

        # ── Save real JFIF image (visual noise image for download) ──
        cipher_image = final_cipher.reshape(original_shape).astype(np.uint8)
        jfif_path    = os.path.join(app.config['OUTPUT_FOLDER'], f"encrypted_{enc_id}.jfif")
        pil_img      = Image.fromarray(cipher_image, mode="RGB")
        pil_img.save(jfif_path, format="JPEG", quality=95)

        orig_b64 = array_to_b64(arr)
        enc_b64  = array_to_b64(cipher_image)

        return jsonify({
            "success" : True,
            "enc_id"  : enc_id,
            "npcr"    : npcr,
            "uaci"    : uaci,
            "orig_b64": orig_b64,
            "enc_b64" : enc_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download_encrypted/<enc_id>")
def download_encrypted(enc_id):
    enc_id = enc_id.upper()
    path   = os.path.join(app.config['OUTPUT_FOLDER'], f"encrypted_{enc_id}.jfif")
    if not os.path.exists(path):
        return jsonify({"error": "Encrypted file not found."}), 404
    return send_file(
        path,
        as_attachment = True,
        download_name = f"encrypted_{enc_id}.jfif",
        mimetype      = "image/jpeg"
    )


@app.route("/decrypt", methods=["POST"])
def decrypt():
    if "jfif_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    jfif_file = request.files["jfif_file"]
    filename  = jfif_file.filename  # e.g. encrypted_FD67A94.jfif

    try:
        # Extract enc_id from filename: encrypted_XXXXXXXX.jfif
        basename = os.path.splitext(filename)[0]          # encrypted_FD67A94
        enc_id   = basename.replace("encrypted_", "").upper()

        # Load lossless data from server
        lossless_path = os.path.join(app.config['OUTPUT_FOLDER'], f"data_{enc_id}.npz")
        if not os.path.exists(lossless_path):
            return jsonify({
                "error": f"Decryption data not found for ID '{enc_id}'. "
                         f"Make sure you upload the .jfif file that was downloaded "
                         f"from this application on the same computer."
            }), 404

        bundle         = np.load(lossless_path)
        cipher_data    = bundle["cipher"]
        original_shape = tuple(bundle["original_shape"])
        T_matrix       = bundle["T_matrix"].astype(int)
        permutation    = bundle["permutation"].astype(int)
        add_key        = int(bundle["add_key"][0])

        # Decrypt
        undiffused  = reverse_diffusion(cipher_data, permutation, add_key)
        chaotic_int = generate_chaotic_sequence(T_matrix, len(undiffused))
        recovered   = bijective_decrypt(undiffused, chaotic_int, T_matrix)

        npcr, uaci  = compute_npcr_uaci(cipher_data, recovered)

        enc_b64 = array_to_b64(cipher_data.reshape(original_shape))
        dec_b64 = array_to_b64(recovered.reshape(original_shape))

        # Save recovered image
        rec_img  = Image.fromarray(recovered.reshape(original_shape).astype(np.uint8), mode="RGB")
        rec_path = os.path.join(app.config['OUTPUT_FOLDER'], "recovered.png")
        rec_img.save(rec_path)

        return jsonify({
            "success": True,
            "npcr"   : npcr,
            "uaci"   : uaci,
            "enc_b64": enc_b64,
            "dec_b64": dec_b64,
        })

    except Exception as e:
        return jsonify({"error": f"Decryption failed: {str(e)}"}), 500


@app.route("/download_recovered")
def download_recovered():
    path = os.path.join(app.config['OUTPUT_FOLDER'], "recovered.png")
    if not os.path.exists(path):
        return jsonify({"error": "No recovered image found. Please decrypt first."}), 404
    return send_file(
        path,
        as_attachment = True,
        download_name = "recovered_image.png",
        mimetype      = "image/png"
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)