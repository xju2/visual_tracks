from omegaconf import OmegaConf

def add_my_resolvers():
    def resolve_if(condition, true_value, false_value):
        return true_value if condition else false_value

    OmegaConf.register_new_resolver("if", resolve_if)

    OmegaConf.register_new_resolver("gen_str",
                                    lambda x, ys: [x.format(y.strip()) for y in ys.split(",")])
    OmegaConf.register_new_resolver("gen_list",
                                    lambda x, y: [x] * y)

    OmegaConf.register_new_resolver("eval", eval)
