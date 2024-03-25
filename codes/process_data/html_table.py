def plot_table(data):
    header = '''<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>table</title>
    </head>
    <body>
    {}
    </body>
</html>'''
    from tabulate import tabulate
    with open("code/process_data/table_plot.html","w") as f:   
        f.write(header.format(tabulate(data,missingval="?",tablefmt="html")))