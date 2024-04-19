"""
python -m display.make_file
"""

# import subprocess
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import DiffLexer

# Generate the diff content using git diff command
# diff_content = subprocess.check_output(["git", "diff", "<commit-id-1>", "<commit-id-2>"]).decode("utf-8")
# read the diff content from a file
with open("/home/chris_cohere_ai/SWE-bench-stuff/display/test1.diff", "r") as file:
    diff_content = file.read()

# Format the diff content as HTML
lexer = DiffLexer()
formatter = HtmlFormatter()
formatted_diff = highlight(diff_content, lexer, formatter)


css = """
<head>
    <style>
        .gd {
            background-color: #f8cbad;
        }
        .gi {
            background-color: #d9ead3;
        }
        .gu {
            background-color: #50524f;
        }

    </style>
</head>
"""

formatted_diff = f"<html>{css}<body>{formatted_diff}</body></html>"


# Save the formatted diff to a file
with open("/home/chris_cohere_ai/SWE-bench-stuff/display/diff.html", "w") as file:
    file.write(formatted_diff)

print ("Diff content has been saved to diff.html")
