import linecache


def extract_method_from_line(java_file_path, line_number):
    with open(java_file_path) as f:
        lines = f.readlines()

    method_start_line = None
    method_end_line = None
    brace_count = 0

    for i, line in enumerate(lines):
        # Check if we've found the line that contains the code smell
        if i + 1 == line_number:
            # Look for the start of the method
            for j in range(i, -1, -1):
                if '}' in lines[j]:
                    brace_count += 1
                elif '{' in lines[j]:
                    brace_count -= 1

                if brace_count == 0:
                    method_start_line = j
                    break

            # Look for the end of the method
            brace_count = 1
            for j in range(i + 1, len(lines)):
                if '{' in lines[j]:
                    brace_count += 1
                elif '}' in lines[j]:
                    brace_count -= 1

                if brace_count == 0:
                    method_end_line = j
                    break

            break

    if method_start_line is None or method_end_line is None:
        return None, None

    method_source_lines = lines[method_start_line:method_end_line + 1]
    method_source = ''.join(method_source_lines)
    return method_source


def extract_line(file_path, line_number):
    return linecache.getline(file_path, line_number)
