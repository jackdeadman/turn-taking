from matplotlib import pyplot as plt


def interactive_scatter(points, chunks):
    import base64
    import io
    import plotly.express as px
    from turntaking.vis import plot_activity, samples_to_activity, plot_number_of_people_speaking
    from turntaking.transcript import perform_activity_reordering

    def create_activity_plot(index):
        info, samples = chunks[index]
        plt.figure()
        bytes_buffer = io.BytesIO()
        plot_activity(
            perform_activity_reordering(samples_to_activity(samples))
        )
        plt.title(info.dataset)
        # plot_number_of_people_speaking(samples_to_activity(samples))
        # plt.ylim(0, 1)
        # Save fig into the buffer
        plt.savefig(bytes_buffer, format='png')
        plt.close('all')
        # Move pointer to the start so it's ready to be read
        bytes_buffer.seek(0)

        base64_binary = base64.b64encode(bytes_buffer.read())
        base64_message = base64_binary.decode('ascii')
        return "data:image/jpg;base64," + base64_message

    fig = px.scatter(
        x=points[:, 0],
        y=points[:, 1]
    )

    import dash
    import dash_core_components as dcc
    import dash_html_components as html

    app = dash.Dash()

    app.layout = html.Div([
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ])

    @app.callback(
        dash.Output("graph-tooltip", "show"),
        dash.Output("graph-tooltip", "bbox"),
        dash.Output("graph-tooltip", "children"),
        dash.Input("graph-basic-2", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, dash.no_update, dash.no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]

        children = [
            html.Div([
                html.Img(src=create_activity_plot(int(num)), style={"width": "100%"})
            ], style={'width': '200px', 'white-space': 'normal'})
        ]

        return True, bbox, children

    app.run_server(debug=True, use_reloader=False)

