
def check_diff(diff):

    if '@@' not in diff:
        return False
    if '---' not in diff:
        return False

    # count the max times any line is repeated in full_output
    lines = diff.split('\n')
    line_count = {}
    for line in lines:
        if line.strip() == '':
            continue
        if line.strip() == '"""':
            continue
        if line in line_count:
            line_count[line] += 1
        else:
            line_count[line] = 1
    max_count = max(line_count.values())
    # rep_counts.append(max_count)

    if max_count > 20:
        return False
    
    return True
            