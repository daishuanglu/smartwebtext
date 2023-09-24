import random
import linecache
import string


def rand_letters(n):
    word_length = random.choice(max(n, 1))
    random_word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
    return random_word


def rand_line_from_file(target_value: str, fpath: str, max_retry=10, **kwargs):
    total_lines = sum(1 for _ in open(fpath, 'r'))
    for _ in range(max_retry):
        rand_line_no = random.randint(1, total_lines)
        rand_line = linecache.getline(fpath, rand_line_no).strip()
        rand_line = rand_line.lower()
        if rand_line != target_value.lower():
            return rand_line
    return rand_letters(20)