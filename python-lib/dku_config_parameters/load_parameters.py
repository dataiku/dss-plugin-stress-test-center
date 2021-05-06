import dataiku
from dataiku.customrecipe import (
    get_recipe_config,
    get_input_names_for_role,
    get_output_names_for_role,
)
from dku_config_parameters.dku_config import DkuConfig


class DkuConfigLoading:
    def __init__(self):
        self.config = get_recipe_config()
        self.dku_config = DkuConfig()


class DkuConfigLoadingStressTestCenter(DkuConfigLoading):
    """Configuration for Ontology Tagging Plugin"""

    ATTACKS_PARAMETERS = [
        "typos",
        "word_swap",
        "word_deletion",
    ]
    ATTACKS_SEVERITY_PARAMETERS = [
        "typos_severity",
        "word_swap_severity",
        "word_deletion_severity",
    ]

    def __init__(self):
        """Instanciate class with DkuConfigLoading and add input datasets to dku_config"""

        super().__init__()
        input_dataset = get_input_names_for_role("input_dataset")[0]
        self.dku_config.add_param(
            name="input_dataset", value=dataiku.Dataset(input_dataset), required=True
        )
        self.input_dataset_columns = [
            p["name"] for p in self.dku_config.input_dataset.read_schema()
        ]
        input_model = get_input_names_for_role("model")[0]
        self.dku_config.add_param(
            name="input_model", value=dataiku.Model(input_model), required=True
        )

    def _content_error_message(self, error, column):
        """Get corresponding error message if any"""

        if error == "missing":
            return "Missing input column."

        if error == "invalid":
            return "Invalid input column : {}.\n".format(column)

    def _add_samples_column(self):
        """Load samples column from Test Dataset"""

        text_column = self.config.get("samples_column")
        input_columns = self.input_dataset_columns
        self.dku_config.add_param(
            name="samples_column",
            value=text_column,
            required=True,
            checks=self._get_column_checks(text_column, input_columns),
        )

    def _get_column_checks(self, column, input_columns):
        """Check for mandatory columns parameters"""

        return [
            {
                "type": "exists",
                "err_msg": self._content_error_message("missing", None),
            },
            {
                "type": "custom",
                "cond": column in input_columns or column == None,
                "err_msg": self._content_error_message("invalid", column),
            },
        ]

    def _add_attacks(self):
        """Load matching parameters"""

        for parameter in self.ATTACKS_PARAMETERS:
            self.dku_config.add_param(
                name=parameter, value=self.config[parameter], required=True
            )

    def _add_severity(self):
        for parameter in self.ATTACKS_SEVERITY_PARAMETERS:
            self.dku_config.add_param(
                name=parameter,
                value=self.config.get(parameter),
                checks=[
                    {
                        "type": "between",
                        "op": (0, 1),
                        "err_msg": "The fraction of the samples to corrupt should be between 0 and 1.",
                    }
                ],
            )

    def _add_output_metrics_dataset(self):
        metrics_output = get_output_names_for_role("metrics_dataset")[0]
        self.dku_config.add_param(
            name="metrics_dataset",
            value=dataiku.Dataset(metrics_output),
            required=True,
        )

    def _add_output_critical_samples_dataset(self):
        critical_samples_output = get_output_names_for_role("critical_samples_dataset")[
            0
        ]
        self.dku_config.add_param(
            name="critical_samples_dataset",
            value=dataiku.Dataset(critical_samples_output),
            required=True,
        )

    def load_settings(self):
        self._add_samples_column()
        self._add_attacks()
        self._add_severity()
        self._add_output_metrics_dataset()
        self._add_output_critical_samples_dataset()
        return self.dku_config
