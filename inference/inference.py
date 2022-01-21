import sys
if sys.version_info < (3, 6):
    class ModuleNotFoundError(Exception):
        pass

import json
import io
try:
    import numpy as np
except (ModuleNotFoundError, ImportError):
    import os
    os.system("pip install numpy")
    import numpy as np

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')
        return d if len(d) else ''

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    if context.request_content_type == "application/x-npy":
        # If we're an array of numpy objects, handle that
        # See https://github.com/aws/sagemaker-python-sdk/issues/799#issuecomment-494564933
        data = np.load(io.BytesIO(data), allow_pickle=True)
        if len(data.shape) is 4:
            data = [x.tolist() for x in data]
        elif len(data.shape) is 3:
            data = data.tolist()
        else:
            raise ValueError("Invalid tensor shape "+str(data.shape))
        return json.dumps({
            "instances": data
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))
    # May need to implement this
    # https://github.com/aws/sagemaker-python-sdk/issues/799#issuecomment-494564933
    # buffer = io.BytesIO()
    # np.save(buffer, data.asnumpy())
    # return buffer.getvalue()
    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
