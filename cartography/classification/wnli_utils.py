from transformers.data.processors.glue import WnliProcessor

class AdaptedWnliProcessor(WnliProcessor):
  def get_examples(self, data_file, set_type):
      return self._create_examples(self._read_tsv(data_file), set_type=set_type)

  def get_labels(self):
      """See base class."""
      return ["not_entailment", "entailment"]