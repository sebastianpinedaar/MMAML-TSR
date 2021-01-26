
import torch
import torch.nn as nn

import sys
sys.path.insert(1, "..")
from metrics import torch_mae as mae

class MetaLearner(object):
    def __init__(self, model, optimizer, fast_lr, loss_func,
                 first_order, num_updates, inner_loop_grad_clip,
                 device):

        self._model = model
        self._fast_lr = fast_lr
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._first_order = first_order
        self._num_updates = num_updates
        self._inner_loop_grad_clip = inner_loop_grad_clip
        self._device = device
        self._grads_mean = []

        self.to(device)


    def update_params(self, loss, params):
        """Apply one step of gradient descent on the loss function `loss`,
        with step-size `self._fast_lr`, and returns the updated parameters.
        """
        create_graph = not self._first_order
        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=create_graph, allow_unused=True)
        for (name, param), grad in zip(params.items(), grads):
            if self._inner_loop_grad_clip > 0 and grad is not None:
                grad = grad.clamp(min=-self._inner_loop_grad_clip,
                                  max=self._inner_loop_grad_clip)
            if grad is not None:
              params[name] = param - self._fast_lr * grad

        return params

    def adapt(self, train_tasks):
        adapted_params = []

        for task in train_tasks:
            params = self._model.param_dict

            for i in range(self._num_updates):
                preds = self._model(task.x, params=params)
                loss = self._loss_func(preds, task.y)
                params = self.update_params(loss, params=params)

            adapted_params.append(params)

        return adapted_params

    def step(self, adapted_params_list, val_tasks,
             is_training, additional_loss_term = None):
        
        self._optimizer.zero_grad()
        post_update_losses = []

        for adapted_params, task in zip(
                adapted_params_list, val_tasks):
            preds = self._model(task.x, params=adapted_params)

            if ~is_training:
                preds = torch.clamp(preds, 0, 1)
                loss = mae(preds, task.y)
            else:
                loss = self._loss_func(preds, task.y)
            post_update_losses.append(loss)

        if additional_loss_term is None:
            mean_loss = torch.mean(torch.stack(post_update_losses))
        else:
            mean_loss = torch.mean(torch.stack(post_update_losses)) + additional_loss_term

        if is_training:
            mean_loss.backward()
            self._optimizer.step()


        return mean_loss

    def to(self, device, **kwargs):
        self._device = device
        self._model.to(device, **kwargs)

    def state_dict(self):
        state = {
            'model_state_dict': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict() 
        }

        return state

    def load_state_dict(self, state_dict):

        self._model.load_state_dict(state_dict["model_state_dict"])
        self._optimizer.load_state_dict(state_dict["optimizer"])


