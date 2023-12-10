from openai import OpenAI


def get_available_openai_models(put_first=None, filter_by=None):
    model_list = OpenAI().models.list()
    model_list = [m.id for m in model_list.data]
    if filter_by:
        model_list = [m for m in model_list if filter_by in m]
    if put_first:
        assert put_first in model_list
        model_list.remove(put_first)
        model_list = [put_first] + model_list
    return model_list
