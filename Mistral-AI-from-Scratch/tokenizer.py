from pathlib import Path
from sentencepiece import SentencePieceProcessor
from typing import List


class Tokenizer:
    """
    一个封装了 SentencePieceProcessor 的分词器类。
    用于将文本字符串与整数令牌列表之间进行转换。
    """
    def __init__(self, model_path: str):
        """
        初始化分词器。
        Args:
            model_path: 指向 SentencePiece 模型文件的路径。
        """
        # 确保模型文件存在
        assert Path(model_path).exists(), model_path
        # 加载 SentencePiece 模型
        self._model = SentencePieceProcessor(model_file=model_path)
        # 验证词汇表大小是否正确
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def n_words(self) -> int:
        """返回词汇表中的令牌数量。"""
        return self._model.vocab_size()

    @property
    def bos_id(self) -> int:
        """返回序列开始（BOS）令牌的 ID。"""
        return self._model.bos_id()

    @property
    def eos_id(self) -> int:
        """返回序列结束（EOS）令牌的 ID。"""
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        """返回填充（PAD）令牌的 ID。"""
        return self._model.pad_id()

    def encode(self, s: str, bos: bool = True) -> List[int]:
        """
        将字符串编码为令牌 ID 列表。
        Args:
            s: 输入的字符串。
            bos: 是否在序列开头添加 BOS 令牌。
        Returns:
            编码后的令牌 ID 列表。
        """
        assert isinstance(s, str)
        # 使用 SentencePiece 模型进行编码
        t = self._model.encode(s)
        # 如果需要，添加 BOS 令牌
        if bos:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: List[int]) -> str:
        """
        将令牌 ID 列表解码为字符串。
        Args:
            t: 令牌 ID 列表。
        Returns:
            解码后的字符串。
        """
        return self._model.decode(t)