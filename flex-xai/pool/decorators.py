"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import functools
from typing import List
import warnings
import copy

from flex.common.utils import check_min_arguments
from flex.model import FlexModel


def ERROR_MSG_MIN_ARG_GENERATOR(f, min_args):
    return f"The decorated function: {f.__name__} is expected to have at least {min_args} argument/s."


def init_server_model(func):
    @functools.wraps(func)
    def _init_server_model_(server_flex_model: FlexModel, _, *args, **kwargs):
        server_flex_model.update(func(*args, **kwargs))

    return _init_server_model_


def deploy_server_model(func):
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

    @functools.wraps(func)
    def _deploy_model_(
        server_flex_model: FlexModel,
        clients_flex_models: List[FlexModel],
        *args,
        **kwargs,
    ):
        for k in clients_flex_models:
            # Reminder, it is not possible to make assignements here
            clients_flex_models[k].update(func(server_flex_model, *args, **kwargs))

    return _deploy_model_


def collect_clients_weights(func):
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

    @functools.wraps(func)
    def _collect_weights_(
        aggregator_flex_model: FlexModel,
        clients_flex_models: List[FlexModel],
        *args,
        **kwargs,
    ):
        if "weights" not in aggregator_flex_model:
            aggregator_flex_model["weights"] = []
        for k in clients_flex_models:
            client_weights = func(clients_flex_models[k], *args, **kwargs)
            aggregator_flex_model["weights"].append(client_weights)

    return _collect_weights_


def aggregate_weights(func):
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

    @functools.wraps(func)
    def _aggregate_weights_(aggregator_flex_model: FlexModel, _, *args, **kwargs):
        aggregator_flex_model["aggregated_weights"] = func(
            aggregator_flex_model["weights"], *args, **kwargs
        )
        aggregator_flex_model["weights"] = []

    return _aggregate_weights_


def set_aggregated_weights(func):
    min_args = 2
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

    @functools.wraps(func)
    def _deploy_aggregated_weights_(
        aggregator_flex_model: FlexModel,
        servers_flex_models: FlexModel,
        *args,
        **kwargs,
    ):
        for k in servers_flex_models:
            func(
                servers_flex_models[k],
                aggregator_flex_model["aggregated_weights"],
                *args,
                **kwargs,
            )

    return _deploy_aggregated_weights_


def evaluate_server_model(func):
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

    @functools.wraps(func)
    def _evaluate_server_model_(server_flex_model: FlexModel, _, *args, **kwargs):
        return func(server_flex_model, *args, **kwargs)

    return _evaluate_server_model_

# flex-xai addition ----------------------------------------------------------------
from flex.data import Dataset

def set_explainer(func):
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )
    
    @functools.wraps(func)
    def _set_explainer_(node_flex_model: FlexModel, *args, **kwargs):
        if "explainers" not in node_flex_model:
            node_flex_model["explainers"] = {}

        name = kwargs.get("name", 'exp')
        # generate a new name if the name is already taken
        if name in node_flex_model["explainers"]:
            base_name = name
            i = 1
            while f"{base_name}_{i}" in node_flex_model["explainers"]:
                i += 1
            name = f"{base_name}_{i}"  
            warnings.warn(f"The explainer '{base_name}' already exists. Renaming to '{name}' to avoid duplicates.", Warning)
            
        exp = func(node_flex_model, *args, **kwargs)
        node_flex_model["explainers"][name] = exp

    return _set_explainer_


def get_explanations(func):
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

    @functools.wraps(func)
    def _get_explanations_(node_flex_model: FlexModel, node_data: Dataset, *args, **kwargs):
        if "explanations" not in node_flex_model:
            node_flex_model["explanations"] = {}
        
        explanations = func(node_flex_model, node_data, *args, **kwargs)
        if explanations is not None:
            node_flex_model["explanations"].update(explanations)

    return _get_explanations_

def centralized(func):
    @functools.wraps(func)
    def _centralized_(servers_flex_model: FlexModel, server_data: Dataset):
        try:
            model = copy.deepcopy(servers_flex_model["model"])
            criterion = copy.deepcopy(servers_flex_model["criterion"])
            optimizer_func = copy.deepcopy(servers_flex_model["optimizer_func"])
            opt_kwargs = copy.deepcopy(servers_flex_model["optimizer_kwargs"])
            explainers = copy.deepcopy(servers_flex_model["explainers"])
        except:
            model = criterion = optimizer_func = opt_kwargs = explainers = None

        
        return model, criterion, optimizer_func, opt_kwargs, explainers, server_data

    return _centralized_


def to_plot_explanation(func):
    @functools.wraps(func)
    def _to_plot_explanation_(node_flex_model: FlexModel, node_data: Dataset, *args, **kwargs):
        dict_result = {} 
        for exp_name, exps in node_flex_model["explanations"].items():
            if (result := func(exps, node_data, name=exp_name, *args, **kwargs)) is not None:
                dict_result[exp_name] = result
        
        return dict_result

    return _to_plot_explanation_