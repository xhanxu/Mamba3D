# from .runner import run_net
from .runner import test_net
from .runner_pretrain import run_net as pretrain_run_net
from .runner_finetune import run_net as finetune_run_net
from .runner_finetune import test_net as test_run_net
from .runner_svm import run_net as svm_run_net
from .runner_fewshot import run_net as fewshot_run_net
from .runner_tsne import tsne_net as tsne_run_net
