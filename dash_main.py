import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc


app = dash.Dash(
    __name__,
    external_stylesheets=["style.css", "bWLwgP.css", "custom_css.css"],
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1(children=["Automl Dashboard"]),
                    width=8,
                    style={"backgroundColor": "#EDEDED"},
                ),
                dbc.Col(html.Img(src="assets/color_schema_.png"), width=4),
            ],
            justify="start",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Label(
                        "Model Name",
                        className="label-color",
                    )
                ),
                dbc.Col(
                    dcc.Input(
                        id="model_name", type="text", placeholder="Enter  Model Name"
                    )
                ),
                dbc.Col(
                    dbc.Label(
                        "Model Id",
                        className="label-color",
                    )
                ),
                dbc.Col(
                    dbc.Input(id="model_id", type="text", placeholder="Enter Model Id")
                ),
            ]
        ),
    ]
)


if __name__ == "__main__":
    app.run_server()
