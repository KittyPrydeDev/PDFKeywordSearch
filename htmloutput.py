from jinja2 import Environment

HTML = """
<html>
<head>
<title>{{ title }}</title>
</head>
<body>
Hello.
</body>
</html>
"""


def print_html_doc():
    output = Environment().from_string(HTML).render(title='Keyword Search Results')
    with open('C:\\tout\sample.html', 'w') as fh:
        fh.write(output)


if __name__ == '__main__':
    print_html_doc()