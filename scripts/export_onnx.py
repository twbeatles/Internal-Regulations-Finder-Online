# -*- coding: utf-8 -*-
"""임베딩 모델 ONNX export + INT8 양자화 스크립트 (경량화용)

기본 백엔드(onnx_fp32)가 실제 ONNX 파일(model.onnx)이 없으면 무거운
torch로 fallback하므로, 이 스크립트로 미리 변환해 두면 CPU 메모리와
추론 속도가 크게 개선됩니다 (E5-small 기준 fp32 torch ~470MB ->
ONNX INT8 ~113MB).

사용법:
    python scripts/export_onnx.py [--model "E5-Small-KO (경량·기본)"] [--int8] [--verify]

    # 특정 모델만 fp32 export
    python scripts/export_onnx.py --model "BGE-M3 (고품질·8192)"

    # INT8 양자화까지 + 검증
    python scripts/export_onnx.py --int8 --verify

필요 패키지:
    pip install optimum[onnxruntime] onnxruntime

참고:
    - Qwen3-Embedding-0.6B 등 디코더형(last-token 풀링) 모델은
      mean-pooling ONNX와 맞지 않아 export를 거부합니다 (torch 전용).
    - Windows 콘솔(cp949)에서 이모지가 깨지면 PYTHONUTF8=1 로 실행하세요.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import AppConfig

INT8_FILENAME = "model.int8.onnx"
FP32_FILENAME = "model.onnx"
VERIFY_THRESHOLD = 0.99


def resolve_model_id(spec: str) -> str:
    """표시 이름 또는 HF ID를 HF ID로 변환. 알 수 없으면 ValueError."""
    models = AppConfig.AVAILABLE_MODELS
    if spec in models:
        return models[spec]
    if spec in models.values():
        return spec
    available = ", ".join(models.keys())
    raise ValueError(f"알 수 없는 모델: {spec}. 사용 가능: {available}")


def ensure_exportable(model_id: str) -> None:
    """디코더형 모델(ONNX mean-pooling 비호환)이면 export 거부."""
    presets = getattr(AppConfig, "EMBED_MODEL_PRESETS", {}) or {}
    if presets.get(model_id, {}).get("force_backend") == "torch":
        raise ValueError(
            f"{model_id}는 디코더형 임베딩이라 ONNX export 대상이 아닙니다 (torch 전용)"
        )


def default_output_dir(model_id: str) -> str:
    return os.path.abspath(
        os.path.join("models", model_id.replace("/", "--"))
    )


def export_fp32(model_id: str, output_dir: str) -> str:
    """HF 모델을 model.onnx(fp32)로 export. optimum 필요."""
    try:
        optimum_ort = __import__("optimum.onnxruntime", fromlist=["ORTModelForFeatureExtraction"])
        transformers_module = __import__("transformers", fromlist=["AutoTokenizer"])
    except ImportError:
        raise RuntimeError("optimum[onnxruntime]가 필요합니다: pip install optimum[onnxruntime]")

    os.makedirs(output_dir, exist_ok=True)
    print(f"🔄 ONNX export 중: {model_id} -> {output_dir}")
    ort_model = optimum_ort.ORTModelForFeatureExtraction.from_pretrained(
        model_id, export=True
    )
    ort_model.save_pretrained(output_dir)
    transformers_module.AutoTokenizer.from_pretrained(model_id).save_pretrained(output_dir)
    onnx_path = os.path.join(output_dir, FP32_FILENAME)
    print(f"   ✅ export 완료: {onnx_path}")
    return onnx_path


def quantize_int8(fp32_path: str, int8_path: str = "") -> str:
    """fp32 onnx를 동적 양자화로 INT8 변환. onnxruntime 필요."""
    try:
        quantization = __import__(
            "onnxruntime.quantization", fromlist=["quantize_dynamic", "QuantType"]
        )
    except ImportError:
        raise RuntimeError("onnxruntime이 필요합니다: pip install onnxruntime")

    dest = int8_path or os.path.join(os.path.dirname(fp32_path), INT8_FILENAME)
    print(f"🔄 INT8 양자화 중: {fp32_path} -> {dest}")
    quantization.quantize_dynamic(
        fp32_path, dest, weight_type=quantization.QuantType.QInt8
    )
    print(f"   ✅ 양자화 완료: {dest}")
    return dest


def _size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def verify_onnx(model_id: str, output_dir: str, onnx_file: str = FP32_FILENAME) -> bool:
    """torch 문장 임베딩과 ONNX 출력의 코사인 유사도로 export 충실도 검증."""
    import numpy as np

    try:
        st_module = __import__("sentence_transformers", fromlist=["SentenceTransformer"])
        transformers_module = __import__("transformers", fromlist=["AutoTokenizer"])
        ort = __import__("onnxruntime")
    except ImportError as e:
        raise RuntimeError(f"검증 패키지 부족: {e}")

    sentences = [
        "연차 휴가는 1년에 15일이 부여됩니다.",
        "보안 정책에 따라 비밀번호를 주기적으로 변경해야 합니다.",
        "회의실 예약은 사전에 시스템을 통해 진행합니다.",
    ]
    print("🔍 ONNX 충실도 검증 중...")
    torch_vecs = np.array(
        st_module.SentenceTransformer(model_id).encode(
            sentences, normalize_embeddings=True
        )
    )

    tokenizer = transformers_module.AutoTokenizer.from_pretrained(output_dir)
    session = ort.InferenceSession(
        os.path.join(output_dir, onnx_file), providers=["CPUExecutionProvider"]
    )
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="np")
    feed = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in [i.name for i in session.get_inputs()] and "token_type_ids" in inputs:
        feed["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
    hidden = session.run(None, feed)[0]
    mask = np.expand_dims(inputs["attention_mask"].astype(np.float32), -1)
    pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
    onnx_vecs = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)

    sims = (torch_vecs * onnx_vecs).sum(axis=1)
    avg_sim = float(sims.mean())
    passed = avg_sim >= VERIFY_THRESHOLD
    print(f"   {'✅' if passed else '⚠️'} 평균 코사인 유사도: {avg_sim:.4f} (기준 {VERIFY_THRESHOLD})")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="임베딩 모델 ONNX export + INT8 양자화")
    parser.add_argument("--model", "-m", default=AppConfig.DEFAULT_MODEL,
                        help="표시 이름 또는 HF ID (기본값: 기본 모델)")
    parser.add_argument("--output", "-o", default=None, help="출력 디렉토리 (기본값: ./models/<org--repo>)")
    parser.add_argument("--int8", action="store_true", help="model.int8.onnx도 생성")
    parser.add_argument("--verify", action="store_true", help="export 충실도 검증")
    parser.add_argument("--skip-export", action="store_true",
                        help="export 생략 (기존 model.onnx로 양자화/검증만)")
    args = parser.parse_args()

    try:
        model_id = resolve_model_id(args.model)
        ensure_exportable(model_id)
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    output_dir = os.path.abspath(args.output) if args.output else default_output_dir(model_id)
    fp32_path = os.path.join(output_dir, FP32_FILENAME)

    try:
        if not args.skip_export:
            export_fp32(model_id, output_dir)
        elif not os.path.exists(fp32_path):
            print(f"❌ 기존 ONNX 없음: {fp32_path}")
            return 1

        print(f"📦 fp32 크기: {_size_mb(fp32_path):.1f} MB")
        ok = True
        if args.int8:
            int8_path = quantize_int8(fp32_path)
            print(f"📦 int8 크기: {_size_mb(int8_path):.1f} MB")
        if args.verify:
            target = INT8_FILENAME if args.int8 else FP32_FILENAME
            ok = verify_onnx(model_id, output_dir, target)
        return 0 if ok else 1
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1
    except Exception as e:  # export/양자화 실패는 그대로 보고
        print(f"❌ 실패: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
