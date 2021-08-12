from transformers.data.processors.glue import RteProcessor

class AdaptedRteProcessor(RteProcessor):
  def get_examples(self, data_file, set_type):
      return self._create_examples(self._read_tsv(data_file), set_type=set_type)