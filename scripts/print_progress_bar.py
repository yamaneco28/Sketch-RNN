import shutil


def print_progress_bar(i, length, width=None, end='\n', header=''):
    digits = len(str(length))
    i_str = format(i+1, '0' + str(digits))
    footer = '{}/{}'.format(i_str, length)
    if width is None:
        terminal_size = shutil.get_terminal_size()
        width = terminal_size.columns-len(header)-len(footer)-5

    if i >= length - 1:
        progress_bar = '=' * width
        end = end
    else:
        num = round(i / (length-1) * width)
        progress_bar = '=' * (num-1) + '>' + ' ' * (width-num)
        end = ''
    print('\r\033[K{} [{}] {}'.format(header, progress_bar, footer), end=end)
