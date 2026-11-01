"""Optional evaluation: compare OCR notes to manually typed correct text."""
from pathlib import Path

def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, letter_a in enumerate(a, 1):
        current = [i]
        for j, letter_b in enumerate(b, 1):
            cost = 0 if letter_a == letter_b else 1
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + cost))
        previous = current
    return previous[-1]

# Put manually typed correct text in reference.txt, and OCR text in predicted.txt.
if Path('reference.txt').exists() and Path('predicted.txt').exists():
    correct = Path('reference.txt').read_text(encoding='utf-8').strip()
    predicted = Path('predicted.txt').read_text(encoding='utf-8').strip()
    if correct:
        cer = edit_distance(correct, predicted) / len(correct)
        print('Character Error Rate:', round(cer * 100, 2), '%')
    else:
        print('reference.txt is empty')
else:
    print('Create reference.txt and predicted.txt to measure OCR error rate.')
