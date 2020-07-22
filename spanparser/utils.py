################################
# Constants

INTENT = 'IN'
SLOT = 'SL'
PAD = '<PAD>'
UNK = '<UNK>'
SOS = '<SOS>'
EOS = '<EOS>'


################################
# Statistics

class Stats(object):
    
    def __init__(self):
        self.n = 0
        self.loss = 0.
        self.accuracy = 0.
        self.grad_norm = 0.

    def add(self, stats):
        """Add another Stats to this one."""
        self.n += stats.n
        self.loss += stats.loss
        self.accuracy += stats.accuracy
        self.grad_norm = max(self.grad_norm, stats.grad_norm)

    def __repr__(self):
        n = max(1, self.n) * 1.
        return '(n={}, loss={:.6f}, accuracy={:.2f}%, grad_norm={:.6f})'.format(
            self.n,
            self.loss / n,
            self.accuracy / n * 100,
            self.grad_norm,
        )

    __str__ = __repr__

    def log(self, tb_logger, step, prefix='', ignore_grad_norm=False):
        """Log to TensorBoard."""
        n = float(self.n)
        tb_logger.add_scalar(prefix + 'loss', self.loss / n, step)
        tb_logger.add_scalar(prefix + 'accuracy', self.accuracy / n * 100, step)
        if not ignore_grad_norm:
            tb_logger.add_scalar(prefix + 'grad_norm', self.grad_norm, step)


################################
# GPU

_GPUS_EXIST = True

def try_gpu(x):
    """Try to put x on a GPU."""
    global _GPUS_EXIST
    if _GPUS_EXIST:
        try:
            return x.cuda()
        except (AssertionError, RuntimeError):
            print('No GPUs detected. Sticking with CPUs.')
            _GPUS_EXIST = False
    return x


def var_to_numpy(v):
    return (v.cpu() if _GPUS_EXIST else v).data.numpy()


################################
# Data

def batch_iter(iterable, batch_size, sort_items_by=None):
    """
    Generates batches of objects of size batch_size each.

    Args:
        iterable: An iterable that yields objects to be batched.
        batch_size (int)
        sort_items_by: (Optional) function mapping objects to keys
            for sorting the batch.
    """
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == batch_size:
            if sort_items_by is not None:
                batch.sort(key=sort_items_by)
            yield batch
            batch = []
    # Last batch
    if batch:
        if sort_items_by is not None:
            batch.sort(key=sort_items_by)
        yield batch


################################
# Unkify

# This function has been adapted from
# https://github.com/clab/rnng/blob/master/get_oracle.py
def unkify(token):
    if len(token.rstrip()) == 0:
        return UNK

    numCaps = 0
    hasDigit = False
    hasDash = False
    hasLower = False
    for char in token.rstrip():
        if char.isdigit():
            hasDigit = True
        elif char == "-":
            hasDash = True
        elif char.isalpha():
            if char.islower():
                hasLower = True
            elif char.isupper():
                numCaps += 1
    result = UNK
    lower = token.rstrip().lower()
    ch0 = token.rstrip()[0]
    if ch0.isupper():
        if numCaps == 1:
            result = result + "-INITC"
        else:
            result = result + "-CAPS"
    elif not (ch0.isalpha()) and numCaps > 0:
        result = result + "-CAPS"
    elif hasLower:
        result = result + "-LC"
    if hasDigit:
        result = result + "-NUM"
    if hasDash:
        result = result + "-DASH"
    if lower[-1] == "s" and len(lower) >= 3:
        ch2 = lower[-2]
        if not (ch2 == "s") and not (ch2 == "i") and not (ch2 == "u"):
            result = result + "-s"
    elif len(lower) >= 5 and not (hasDash) and not (hasDigit and numCaps > 0):
        if lower[-2:] == "ed":
            result = result + "-ed"
        elif lower[-3:] == "ing":
            result = result + "-ing"
        elif lower[-3:] == "ion":
            result = result + "-ion"
        elif lower[-2:] == "er":
            result = result + "-er"
        elif lower[-3:] == "est":
            result = result + "-est"
        elif lower[-2:] == "ly":
            result = result + "-ly"
        elif lower[-3:] == "ity":
            result = result + "-ity"
        elif lower[-1] == "y":
            result = result + "-y"
        elif lower[-2:] == "al":
            result = result + "-al"
    return result
