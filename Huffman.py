# coding=utf8
import heapq
import io
import itertools
import os
import struct
import sys
import typing as t
from collections import Counter, deque
from enum import Enum

DECODER_FAIL = 1
DECODER_COMPLETE = (1 << 1)
PSEUDO_EOF = chr(4)  # End Of Transmission
class Direction(Enum):
    """Code values for directions (left or right) in Huffman tree."""
    LEFT = 0
    RIGHT = 1
class TreeNode:
    def __init__(
        self,
        char: str = "",
        freq: int = 0,
        left: t.Optional["TreeNode"] = None,
        right: t.Optional["TreeNode"] = None,
    ) -> None:
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
        self.fsm_state: t.Optional[int] = None

    def __lt__(self, other):
        return self.freq < other.freq

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        # Wrap self.char in repr() as it could be "\n" or similar.
        return (
            "TreeNode("
                f"char={repr(self.char)}, freq={self.freq},"
                f" left={self.left}, right={self.right}"
            ")"
        )

class HuffmanTree:
    def __init__(self, frequency_table: t.Optional[t.Dict[str, int]] = None):
        self.root: t.Optional[TreeNode] = None
        if frequency_table:
            self.root = self.build_tree(frequency_table)

    def build_tree(self, frequency_table: t.Dict[str, int]) -> t.Optional[TreeNode]:
        heap = [TreeNode(char, freq) for char, freq in frequency_table.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = TreeNode(
                char="",
                freq=left.freq + right.freq,
                left=left,
                right=right,
            )
            heapq.heappush(heap, merged)

        return heap[0] if heap else None

    def generate_codes(self) -> t.Dict[str, t.Tuple[int, int]]:
        def _generate_codes_helper(node: TreeNode, path: int, length: int):
            if node.left is None and node.right is None:
                encoding_table[node.char] = (length, path)
                return
            if node.left is not None:
                _generate_codes_helper(node.left, (path << 1) | 0, length + 1)
            if node.right is not None:
                _generate_codes_helper(node.right, (path << 1) | 1, length + 1)

        encoding_table: t.Dict[str, t.Tuple[int, int]] = {}
        if self.root is not None:
            _generate_codes_helper(self.root, 0, 0)
        return encoding_table

    @staticmethod
    def from_encoding_table(encoding_table: t.Dict[str, t.Tuple[int, int]]) -> TreeNode:
        root = TreeNode()

        for char, (length, code) in encoding_table.items():
            current = root
            for i in range(length):
                bit = (code >> (length - 1 - i)) & 1
                if bit == 0:
                    if current.left is None:
                        current.left = TreeNode()
                    current = current.left
                else:
                    if current.right is None:
                        current.right = TreeNode()
                    current = current.right
            current.char = char

        return root
def generate_fsm_decoder(
    root: TreeNode,
    word_size: int = 4,
) -> list[tuple[int, int, str]]:
    if word_size not in [1, 2, 4, 8]:
        raise ValueError("Possible values for word_size are: 1, 2, 4 or 8.")

    # Create all states for the FSM. This can't be done dynamically in
    # the FSM creation algorithm, because transitions could take you to
    # states that aren't visited yet (in a BFS manner).
    state = 0
    q: t.Deque[TreeNode] = deque([root])
    while q:
        node = q.popleft()

        if node.left is not None:
            q.append(node.left)
        if node.right is not None:
            q.append(node.right)

        # In the FSM, only internal Huffman tree nodes will becomes
        # states.
        if node.left is None and node.right is None:
            continue
        else:
            node.fsm_state = state
            state += 1


    def get_individual_bits(num: int) -> t.Generator[t.Literal[0, 1], None, None]:
        """Get individal bits, left to right, from a given number.

        For example, given `num=6` and `word_size=4`, then the binary
        representation would be `0110` and thus the individual bits are
        `0` -> `1` -> `1` -> `0`.

        """
        nonlocal word_size

        for bit_idx in range(word_size-1, -1, -1):
            yield num & (1 << bit_idx)  # type: ignore


    # Generate all transitions for the FSM, where a transition is
    # defined as:
    #   (state_after_transition, is_invalid_transition, to_emit)
    fsm = []
    q: t.Deque[TreeNode] = deque([root])
    # 所有可能的转换数 例如2(00 01 10 11)
    num_transitions = 1 << word_size
    while q:
        #获取节点的子内部节点
        node = q.popleft()

        if node.left is not None:
            q.append(node.left)
        if node.right is not None:
            q.append(node.right)

        # Only consider internal nodes of the tree.
        if node.left is None and node.right is None:
            continue
        # 遍历所有可能的转换 例如(00 01 10 11)
        inner = []
        for transition in range(num_transitions):
            # 初始化当前节点为 node
            node_after_transition = node
            # 用于存储将要输出的字符
            to_emit = ""
            # 用于存储解码标志。
            flags = 0
            # 将当前转换值转换为单个比特值的序列 num=6 word_size=4 0110 , 16 ws 4 1111
            # 寻找next
            for bit_value in get_individual_bits(transition):
                if bit_value == Direction.LEFT.value:
                    node_after_transition = node_after_transition.left
                else:
                    node_after_transition = node_after_transition.right

                # The given transition can't be done from the current
                # state. This could happen if the FSM is asked to decode
                # an invalid sequence of bytes, i.e. one that isn't
                # encoded using the given Huffman tree.
                if node_after_transition is None:
                    flags |= DECODER_FAIL
                    break

                # Encountered a leaf node, so transition back to root.
                if (
                    node_after_transition.left is None
                    and node_after_transition.right is None
                ):
                    if node_after_transition.char == PSEUDO_EOF:
                        flags |= DECODER_COMPLETE
                        # Once the PSEUDO_EOF is read, we no longer
                        # decode. If we don't, then `to_emit` could
                        # contain characters that weren't in the
                        # encoded text.
                        break
                    else:
                        to_emit += node_after_transition.char

                    node_after_transition = root

            # Add transition to FSM.
            if flags & DECODER_FAIL:
                inner.append((None, flags, to_emit))
            else:
                inner.append(
                    (
                        node_after_transition.fsm_state,  # type: ignore
                        flags,
                        to_emit
                    )
                )
        fsm.append(inner)

    return fsm


def generate_fsm_decoder_by_table(
        huffman_dict,
        word_size: int = 4,
) -> list[tuple[int, int, str]]:
    if word_size not in [1, 2, 4, 8]:
        raise ValueError("Possible values for word_size are: 1, 2, 4 or 8.")

    tree = HuffmanTree.from_encoding_table(huffman_dict)
    state = 0
    q: t.Deque[TreeNode] = deque([tree])
    # 给每个节点编号
    while q:
        node = q.popleft()

        if node.left is not None:
            q.append(node.left)
        if node.right is not None:
            q.append(node.right)

        # In the FSM, only internal Huffman tree nodes will becomes
        # states.
        if node.left is None and node.right is None:
            continue
        else:
            node.fsm_state = state
            state += 1

    def get_individual_bits(num: int) -> t.Generator[t.Literal[0, 1], None, None]:
        """Get individal bits, left to right, from a given number.

        For example, given `num=6` and `word_size=4`, then the binary
        representation would be `0110` and thus the individual bits are
        `0` -> `1` -> `1` -> `0`.

        """
        nonlocal word_size

        for bit_idx in range(word_size - 1, -1, -1):
            yield num & (1 << bit_idx)  # type: ignore

        # Generate all transitions for the FSM, where a transition is
        # defined as:
        #   (state_after_transition, is_invalid_transition, to_emit)

    fsm = []
    q: t.Deque[TreeNode] = deque([tree])
    # 每个节点所有可能的转换数 例如2(00 01 10 11)
    num_transitions = 1 << word_size
    # bfs遍历
    while q:
        # 获取节点的子内部节点
        node = q.popleft()

        if node.left is not None:
            q.append(node.left)
        if node.right is not None:
            q.append(node.right)

        # 不考虑叶子节点
        if node.left is None and node.right is None:
            continue
        # 遍历内部节点所有可能的转换 例如(00 01 10 11)
        inner = []
        for transition in range(num_transitions):
            # 初始化当前节点为 node
            node_after_transition = node
            # 用于存储将要输出的字符
            to_emit = ""
            # 用于存储解码标志。
            flags = 0
            # 将当前转换值转换为单个比特值的序列 num=6 word_size=4 0110 , 16 ws 4 1111
            # 寻找next
            for bit_value in get_individual_bits(transition):
                if bit_value == Direction.LEFT.value:
                    node_after_transition = node_after_transition.left
                else:
                    node_after_transition = node_after_transition.right

                # The given transition can't be done from the current
                # state. This could happen if the FSM is asked to decode
                # an invalid sequence of bytes, i.e. one that isn't
                # encoded using the given Huffman tree.
                if node_after_transition is None:
                    flags |= DECODER_FAIL
                    break

                # Encountered a leaf node, so transition back to root.
                if (
                        node_after_transition.left is None
                        and node_after_transition.right is None
                ):
                    if node_after_transition.char == PSEUDO_EOF:
                        flags |= DECODER_COMPLETE
                        # Once the PSEUDO_EOF is read, we no longer
                        # decode. If we don't, then `to_emit` could
                        # contain characters that weren't in the
                        # encoded text.
                        break
                    else:
                        to_emit += node_after_transition.char

                    node_after_transition = tree

            # Add transition to FSM.
            if flags & DECODER_FAIL:
                inner.append((None, flags, to_emit))
            else:
                inner.append(
                    (
                        node_after_transition.fsm_state,  # type: ignore
                        flags,
                        to_emit
                    )
                )
        fsm.append(inner)

    return fsm
def _get_huffman_tree(freq_table: dict[str, int]) -> TreeNode:
    """Constructs a Huffman tree."""
    heap = []
    for char in freq_table:
        node = TreeNode(char=char, freq=freq_table[char])
        heap.append(node)

    heapq.heapify(heap)
    while len(heap) != 1:
        least_freq1, least_freq2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(
            heap,
            TreeNode(
                freq=least_freq1.freq + least_freq2.freq,
                left=least_freq1,
                right=least_freq2,
            ),
        )

    # Return root node.
    return heap[0]


freq_table = {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45, 'g': 23, 's': 23, 'q': 42}
huffman_tree = _get_huffman_tree(freq_table)
huffman_tree2 = HuffmanTree(freq_table).generate_codes()
rebuilt_tree = HuffmanTree.from_encoding_table(huffman_tree2)
print(huffman_tree)
print(rebuilt_tree)
print(huffman_tree2)
fsm = generate_fsm_decoder(huffman_tree, word_size=2)
fsm2 = generate_fsm_decoder(rebuilt_tree, word_size=2)
print(fsm)
print(fsm2)

fsm = generate_fsm_decoder_by_table(huffman_tree2, word_size=2)
print(fsm)
