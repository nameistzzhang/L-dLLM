import logging
from datasets import load_from_disk, concatenate_datasets

# ==========================================
# 0. 日志配置
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. 路径配置 (请替换为你的实际物理路径)
# ==========================================
# 包含 SFT 和 Starcoder2 的旧文件夹
DIR_SFT_CODE = "/scratch/aszalay1/tianze/cpt_data/18b_pyarrow/" 

# 包含新跑出来的 Fineweb-edu 文件夹
DIR_FINEWEB = "/scratch/aszalay1/tianze/cpt_data/fineweb_pyarrow/" 

# 最终合并后输出的超级数据包文件夹
FINAL_OUTPUT_DIR = "/scratch/aszalay1/tianze/cpt_data/cpt_final_pyarrow/"

def main():
    # ==========================================
    # 2. 极速加载 (内存映射，秒开)
    # ==========================================
    logger.info(f"📥 Loading SFT + Starcoder data from {DIR_SFT_CODE}...")
    ds_sft_code = load_from_disk(DIR_SFT_CODE)
    logger.info(f"   Size: {len(ds_sft_code):,} chunks")

    logger.info(f"📥 Loading FineWeb-edu data from {DIR_FINEWEB}...")
    ds_fineweb = load_from_disk(DIR_FINEWEB)
    logger.info(f"   Size: {len(ds_fineweb):,} chunks")

    # ==========================================
    # 3. 强力拼接
    # ==========================================
    logger.info("🔗 Concatenating datasets...")
    # concatenate_datasets 在 Arrow 层级只是把 metadata 连起来，瞬间完成
    combined_ds = concatenate_datasets([ds_sft_code, ds_fineweb])
    logger.info(f"   Combined size: {len(combined_ds):,} chunks")

    # ==========================================
    # 4. 全局大混洗 (CPT 训练的命门！)
    # ==========================================
    logger.info("🔀 Performing Global Shuffle (seed=42)...")
    # 这一步会在硬盘上重新排列 Arrow 的数据块索引
    shuffled_ds = combined_ds.shuffle(seed=42)

    # ==========================================
    # 5. 落盘保存
    # ==========================================
    logger.info(f"💾 Saving final mixed & shuffled dataset to {FINAL_OUTPUT_DIR}...")
    # 这一步会实际把混洗后的数据物理写入新的文件夹
    shuffled_ds.save_to_disk(FINAL_OUTPUT_DIR)
    
    logger.info("✅ Done! The ultimate Flow Matching CPT dataset is ready.")

if __name__ == "__main__":
    main()