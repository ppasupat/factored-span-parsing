"""
Simple HTTP server for prediction.

Bottle does not play well with classes, so let's use plain methods

Requirements:
- bottle
- the Dataset object must implement
  - s_parse_request(q) --> batch (list[Example])
    where q = The request's forms MultiDict
    (https://bottlepy.org/docs/dev/api.html#bottle.BaseRequest.forms)
  - s_generate_response(q, batch, logit, prediction) --> any
"""
from __future__ import (absolute_import, division, print_function)
from bottle import post, request, run


@post('/pred')
def process():
    q = request.forms
    batch = EXP.dataset.s_parse_request(q)
    logit = EXP.model(batch)
    prediction = EXP.model.get_pred(logit, batch)
    return EXP.dataset.s_generate_response(q, batch, logit, prediction)


def start_server(exp, port):
    global EXP
    EXP = exp
    # This will open a global port!
    # Feel free to use a better server backend than wsgiref.
    # https://bottlepy.org/docs/dev/deployment.html
    run(server='wsgiref', host='0.0.0.0', port=port)
    print('\nGood bye!')
