from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class Callbacks(TrainerCallback):
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(control)
        #control.should_evalute = False
        #control.should_log = False
        return super().on_epoch_begin(args, state, control, **kwargs)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(control)
        return super().on_step_end(args, state, control, **kwargs)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_evalute = False
        control.should_log = False
        return super().on_epoch_end(args, state, control, **kwargs)

