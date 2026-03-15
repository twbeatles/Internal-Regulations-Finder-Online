# -*- coding: utf-8 -*-
"""
모델 사전 다운로드 스크립트 (오프라인/폐쇄망 환경용)

사용법:
    python download_models.py [--output ./models] [--model MODEL_NAME]
    
예시:
    # 모든 모델 다운로드 (기본 ./models 폴더)
    python download_models.py
    
    # 특정 경로에 다운로드
    python download_models.py --output D:/offline_models
    
    # 특정 모델만 다운로드
    python download_models.py --model "SNU SBERT (고성능)"
    
다운로드 완료 후:
    1. 생성된 models 폴더를 폐쇄망 서버로 복사
    2. settings.json에서 offline_mode: true 설정
    3. local_model_path를 모델 폴더 경로로 설정 (선택사항)
"""

import os
import sys
import argparse
import time
import importlib
from typing import Dict, Optional

# 사용 가능한 모델 목록 (server.py의 AppConfig.AVAILABLE_MODELS와 동일)
AVAILABLE_MODELS: Dict[str, str] = {
    "SNU SBERT (고성능)": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    "BM-K Simal (균형)": "BM-K/ko-simal-roberta-base",
    "JHGan SBERT (빠름)": "jhgan/ko-sbert-nli"
}


def print_header():
    """헤더 출력"""
    print("=" * 60)
    print("🚀 사내 규정 검색기 - 모델 다운로드 도구")
    print("   (오프라인/폐쇄망 환경 사전 준비용)")
    print("=" * 60)


def print_models():
    """사용 가능한 모델 목록 출력"""
    print("\n📋 사용 가능한 모델:")
    for i, (name, model_id) in enumerate(AVAILABLE_MODELS.items(), 1):
        print(f"   {i}. {name}")
        print(f"      └── {model_id}")
    print()


def download_model(model_name: str, model_id: str, output_dir: str) -> bool:
    """단일 모델 다운로드
    
    Args:
        model_name: 모델 표시 이름
        model_id: HuggingFace 모델 ID
        output_dir: 다운로드 경로
        
    Returns:
        성공 여부
    """
    print(f"\n🔄 다운로드 중: {model_name}")
    print(f"   모델 ID: {model_id}")
    print(f"   저장 경로: {output_dir}")
    
    try:
        # sentence-transformers 사용 (가장 안정적인 다운로드 방법)
        sentence_transformers_module = importlib.import_module('sentence_transformers')
        sentence_transformer_cls = getattr(sentence_transformers_module, 'SentenceTransformer')
        
        start_time = time.time()
        
        # 모델 다운로드 및 로드 (캐시 폴더 지정)
        model = sentence_transformer_cls(
            model_id,
            cache_folder=output_dir
        )
        
        # 모델 저장 (명시적 저장)
        model_save_path = os.path.join(output_dir, model_id.replace('/', '--'))
        model.save(model_save_path)
        
        elapsed = time.time() - start_time
        print(f"   ✅ 완료! ({elapsed:.1f}초)")
        
        # 모델 폴더 크기 계산
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_save_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
        
        size_mb = total_size / (1024 * 1024)
        print(f"   📦 모델 크기: {size_mb:.1f} MB")
        
        # 메모리 정리
        del model
        import gc
        gc.collect()
        
        try:
            torch = importlib.import_module('torch')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        return True
        
    except ImportError as e:
        print(f"   ❌ 라이브러리 오류: {e}")
        print(f"      sentence-transformers 설치 필요: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"   ❌ 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_model(model_id: str, output_dir: str) -> bool:
    """다운로드된 모델 검증
    
    Args:
        model_id: HuggingFace 모델 ID
        output_dir: 모델 경로
        
    Returns:
        검증 성공 여부
    """
    model_path = os.path.join(output_dir, model_id.replace('/', '--'))
    
    # 필수 파일 확인
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
    alt_files = ['model.safetensors']  # 대체 파일 (newer format)
    
    if not os.path.exists(model_path):
        return False
    
    # 최소한 config.json은 있어야 함
    if not os.path.exists(os.path.join(model_path, 'config.json')):
        return False
    
    # 모델 가중치 파일 확인 (pytorch_model.bin 또는 model.safetensors)
    has_weights = (
        os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) or
        os.path.exists(os.path.join(model_path, 'model.safetensors'))
    )
    
    return has_weights


def main():
    parser = argparse.ArgumentParser(
        description='사내 규정 검색기 - 모델 사전 다운로드 도구'
    )
    parser.add_argument(
        '--output', '-o',
        default='./models',
        help='모델 저장 경로 (기본값: ./models)'
    )
    parser.add_argument(
        '--model', '-m',
        default=None,
        help='특정 모델만 다운로드 (예: "SNU SBERT (고성능)")'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='사용 가능한 모델 목록 출력'
    )
    parser.add_argument(
        '--verify', '-v',
        action='store_true',
        help='기존 다운로드 모델 검증만 수행'
    )
    
    args = parser.parse_args()
    
    print_header()
    
    if args.list:
        print_models()
        return 0
    
    # 출력 디렉토리 생성
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 출력 경로: {output_dir}")
    
    # 다운로드할 모델 결정
    if args.model:
        if args.model not in AVAILABLE_MODELS:
            print(f"\n❌ 알 수 없는 모델: {args.model}")
            print_models()
            return 1
        models_to_download = {args.model: AVAILABLE_MODELS[args.model]}
    else:
        models_to_download = AVAILABLE_MODELS
    
    # 검증만 수행
    if args.verify:
        print("\n🔍 모델 검증 중...")
        all_ok = True
        for name, model_id in models_to_download.items():
            if verify_model(model_id, output_dir):
                print(f"   ✅ {name}: 정상")
            else:
                print(f"   ❌ {name}: 없음 또는 불완전")
                all_ok = False
        
        if all_ok:
            print("\n✅ 모든 모델 검증 완료!")
        else:
            print("\n⚠️ 일부 모델이 없거나 불완전합니다.")
            print("   다운로드를 실행하려면 --verify 옵션을 제거하고 다시 실행하세요.")
        return 0 if all_ok else 1
    
    # 다운로드 실행
    print(f"\n📥 다운로드할 모델: {len(models_to_download)}개")
    print_models() if not args.model else None
    
    # 의존성 확인
    print("\n🔍 의존성 확인 중...")
    try:
        torch = importlib.import_module('torch')
        print(f"   ✅ PyTorch {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch가 설치되어 있지 않습니다.")
        print("      pip install torch")
        return 1
    
    try:
        sentence_transformers = importlib.import_module('sentence_transformers')
        print(f"   ✅ sentence-transformers {sentence_transformers.__version__}")
    except ImportError:
        print("   ❌ sentence-transformers가 설치되어 있지 않습니다.")
        print("      pip install sentence-transformers")
        return 1
    
    # 다운로드 시작
    print("\n" + "=" * 60)
    print("📥 모델 다운로드 시작")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    
    for name, model_id in models_to_download.items():
        if download_model(name, model_id, output_dir):
            success_count += 1
        else:
            fail_count += 1
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 다운로드 결과")
    print("=" * 60)
    print(f"   ✅ 성공: {success_count}개")
    print(f"   ❌ 실패: {fail_count}개")
    print(f"   📂 저장 위치: {output_dir}")
    
    if fail_count == 0:
        print("\n" + "=" * 60)
        print("🎉 모든 모델 다운로드 완료!")
        print("=" * 60)
        print("\n📋 다음 단계:")
        print(f"   1. {output_dir} 폴더를 폐쇄망 서버로 복사")
        print("   2. config/settings.json에서 다음 설정:")
        print('      "offline_mode": true')
        print(f'      "local_model_path": "{output_dir}"')
        print("   3. 서버 재시작")
    else:
        print("\n⚠️ 일부 모델 다운로드에 실패했습니다.")
        print("   네트워크 연결을 확인하고 다시 시도하세요.")
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
