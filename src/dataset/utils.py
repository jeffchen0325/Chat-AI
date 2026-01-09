from pathlib import Path
import json
from typing import Callable

def get_absolute_path(relative_path, module_obj=None)->str:
    """
    获取相对路径的绝对路径

    参数:
        relative_path: 相对路径字符串
        module_obj: 参考模块对象，用于确定基准目录（默认为当前文件所在目录的父目录）
        check_exists: 是否检查路径是否存在

    返回:
        Path: 绝对路径对象

    异常:
        FileNotFoundError: 当check_exists=True且路径不存在时
    """
    # 确定基准目录
    if module_obj is not None:
        # 使用模块文件所在目录作为基准目录
        module_dir = Path(module_obj.__file__).parent
    else:
        # 默认使用当前文件所在目录的父目录
        module_dir = Path(__file__).parent.parent

    # 构建绝对路径
    absolute_path = module_dir / relative_path
    absolute_path = absolute_path.resolve()

    # 检查路径是否存在
    if not absolute_path.exists():
        raise FileNotFoundError(f"路径不存在: {absolute_path}")

    return str(absolute_path)


class IDGenerator:
    """ID生成器"""

    @staticmethod
    def sequential_starting_from(start_id: int) -> Callable[[], int]:
        """从指定数字开始的顺序ID生成器"""
        counter = start_id - 1

        def generator():
            nonlocal counter
            counter += 1
            return counter

        return generator

    @staticmethod
    def calculate_next_id(filepath: str | Path) -> Callable[[], int]:
        """
        从现有JSON文件中计算下一个ID

        参数:
            filepath: JSON文件路径
        """
        filepath = Path(filepath)

        def generator():
            try:
                # 如果文件不存在，从1开始
                if not filepath.exists():
                    return 1

                # 读取JSON文件
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 从labels中获取最大的ID
                labels = data.get('labels', [])
                if not labels:
                    return 1

                # 找到最大的ID
                max_id = max(label.get('id', 0) for label in labels)
                return max_id + 1

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"❌ 读取ID失败，从1开始: {e}")
                return 1

        return generator

# 使用示例
if __name__ == "__main__":
    from configs import config as cfg

    try:
        # 使用模块对象
        audio_path = get_absolute_path(cfg.test_audio, module_obj=cfg)
        print(f"✅ 音频路径: {audio_path}")

    except FileNotFoundError as e:
        print(f"❌ 文件不存在: {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")