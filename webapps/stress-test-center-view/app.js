const webAppConfig = dataiku.getWebAppConfig();
const modelId = webAppConfig['modelId'];
const versionId = webAppConfig['versionId'];

(function() {
    'use strict';
    app.service("CorruptionUtils", function(MetricNames) {
        function toLowerCaseIfNotAcronym(string) {
            if (string === string.toUpperCase()) return string;
            return string.toLowerCase();
        }

        function perfMetricDescription(perfMetric, baseMetric, isRegression) {
            const longName = baseMetric && toLowerCaseIfNotAcronym(MetricNames.longName(baseMetric));
            switch(perfMetric) {
            case "perf_var":
                return ` is the difference in the model's ${longName} between the altered ` +
                    "dataset and the unaltered one.";
            case "corruption_resilience":
                if (isRegression) {
                    return " is the ratio of rows where the corruption does not increase " +
                        "the prediction error.";
                }
                return " is the ratio of rows where the prediction is not altered after " +
                    "the corruption.";
            case "worst_subpop_perf":
                return ` is the worst-case ${longName} across all the modalities of a ` +
                    "categorical feature.";
            default:
                return null;
            }
        }

        function perfMetricName(perfMetric, baseMetric, longer) {
            if (perfMetric === "corruption_resilience") return "Corruption resilience";
            const name = longer ? MetricNames.longName(baseMetric) : MetricNames.shortName(baseMetric);
            switch(perfMetric) {
                case "perf_before":
                return `${name} before`;
            case "perf_after":
                return `${name} after`;
            case "perf_var":
                return `${name} variation`;
            case "worst_subpop_perf":
                return `Worst subpopulation ${toLowerCaseIfNotAcronym(name)}`;
            default:
                return null;
            }
        }

        return {
            perfMetric: {
                name: perfMetricName,
                description: perfMetricDescription,
                isContextual: (perfMetric) => ["perf_before", "perf_after"].includes(perfMetric)
            },
            TYPES: {
                FEATURE_PERTURBATION: {
                    displayName: "Feature corruptions",
                    description: "Each of these independent tests corrupts one or several features across randomly sampled rows."
                },
                TARGET_SHIFT: {
                    displayName: "Target distribution shift",
                    description: "This stress test resamples the test set to match the desired distribution for the target column."
                },
                SUBPOPULATION_SHIFT: {
                    displayName: "Feature distribution shift",
                    description: "This stress test resamples the test set to match the desired distribution for the selected categorical feature."
                }
            },
            TEST_CONSTRAINTS: {
                RebalanceTarget: {
                    allowedPredictionTypes: ["BINARY_CLASSIFICATION", "MULTICLASS"]
                },
                RebalanceFeature: {
                    allowedFeatureTypes: ["CATEGORY"]
                },
                MissingValues: {
                    allowedFeatureTypes: ["NUMERIC", "CATEGORY", "VECTOR", "TEXT"]
                },
                Scaling: {
                    allowedFeatureTypes: ["NUMERIC"]
                }
            }
        };
    });

    app.service('MetricNames', function() { 
        const metrics = {
            F1: {
                longName: "F1 score",
                predType: ["BINARY_CLASSIFICATION", "MULTICLASS"]
            },
            ACCURACY: {
                shortName: "Accuracy",
                predType: ["BINARY_CLASSIFICATION", "MULTICLASS"]
            },
            PRECISION: {
                shortName: "Precision",
                predType: ["BINARY_CLASSIFICATION", "MULTICLASS"]
            },
            RECALL: {
                shortName: "Recall",
                predType: ["BINARY_CLASSIFICATION", "MULTICLASS"]
            },
            COST_MATRIX: {
                shortName: "Cost matrix gain",
                predType: ["BINARY_CLASSIFICATION"]
            },
            ROC_AUC: {
                shortName: "AUC",
                predType: ["BINARY_CLASSIFICATION", "MULTICLASS"]
            },
            LOG_LOSS: {
                shortName: "Log loss",
                predType: ["BINARY_CLASSIFICATION", "MULTICLASS"]
            },
            CUMULATIVE_LIFT: {
                longName: "Cumulative lift",
                shortName: "Lift",
                predType: ["BINARY_CLASSIFICATION"]
            },
            EVS: {
                longName: "Explained variance score",
                predType: ["REGRESSION"]
            },
            MAPE: {
                longName: "Mean absolute percentage error",
                predType: ["REGRESSION"]
            },
            MAE: {
                longName: "Mean absolute error",
                predType: ["REGRESSION"]
            },
            MSE: {
                longName: "Mean squared error",
                predType: ["REGRESSION"]
            },
            RMSE: {
                longName: "Root mean square error",
                predType: ["REGRESSION"]
            },
            RMSLE: {
                longName: "Root mean square logarithmic error",
                predType: ["REGRESSION"]
            },
            R2: {
                longName: "R2 score",
                predType: ["REGRESSION"]
            }
        }
        return {
            availableMetrics: function(predType) {
                return Object.keys(metrics).filter(_ => metrics[_].predType.includes(predType));
            },
            rawNames: Object.keys(metrics),
            longName: metric => metrics[metric].longName || metrics[metric].shortName,
            shortName: metric => metrics[metric].shortName || metric
        }
    });

    app.controller('VizController', function($scope, $http, ModalService, CorruptionUtils,
        MetricNames, $filter) {

        $scope.modal = {};
        $scope.removeModal = ModalService.remove($scope.modal);
        $scope.createModal = ModalService.create($scope.modal);

        $scope.CORRUPTION_TYPES = CorruptionUtils.TYPES;
        $scope.perfMetric = CorruptionUtils.perfMetric;
        $scope.displayWithuserFriendlyMetricName = function(str) {
            if (!str) return;
            const pattern = new RegExp(MetricNames.rawNames.join("|"), "g");
            return str.replace(pattern, matched =>  MetricNames.longName(matched);
        };

        $scope.loading = {};
        $scope.forms = {};

        $scope.settings = {
            tests: {
                RebalanceTarget: {
                    params: { priors: {} }
                },
                RebalanceFeature: {
                    params: { priors: {} }
                },
                MissingValues: {
                    params: { samples_fraction: .5 },
                    selected_features: new Set()
                },
                Scaling: {
                    params: {
                        samples_fraction: .5,
                        scaling_factor: 10
                    },
                    selected_features: new Set()
                }
            },
            samples: .8,
            randomSeed: 1337
        };

        $scope.modelInfo = {
            featureCategories: {}
        };

        $scope.uiState = {
            _dku_stress_test_uncorrupted: {
                displayName: "No corruption"
            },
            RebalanceTarget: {
                displayName: "Shift target distribution"
            },
            RebalanceFeature: {
                displayName: "Shift feature distribution"
            },
            MissingValues: {
                displayName: "Insert missing values"
            },
            Scaling: {
                displayName: "Multiply by a coefficient"
            }
        }

        const featureTypesToIconClass = {
            NUMERIC: "numerical",
            CATEGORY: "icon-font",
            TEXT: "icon-italic",
            VECTOR: "vector"
        };

        $scope.featureToTypeIcon = function(feature) {
            return featureTypesToIconClass[features[feature]];
        }

        $scope.canRunTests = function() {
            return $scope.forms.settings.$valid
                && Object.values($scope.uiState).some(t => t.activated);
        }

        $scope.getFeatureCategories = function(feature) {
            if (!$scope.modelInfo.featureCategories[feature]) {
                $scope.loading.featureCategories = true;
                $http.get(getWebAppBackendUrl(feature + "/categories")).then(function(response) {
                    $scope.loading.featureCategories = false;
                    $scope.modelInfo.featureCategories[feature] = response.data;
                }, function(e) {
                    $scope.loading.featureCategories = false;
                    $scope.createModal.error(e.data);
                });
            }
            $scope.settings.tests.RebalanceFeature.params.priors = {};
        }

        $scope.runAnalysis = function () {
            if (!$scope.canRunTests()) return;
            const requestParams = Object.assign({}, $scope.settings);
            requestParams.tests = {};
            angular.forEach($scope.settings.tests, function(testParams, testName) {
                if (!$scope.uiState[testName].activated) return;
                requestParams.tests[testName] = Object.assign({}, testParams);
                if (testParams.selected_features) { // test is a sample perturbation
                    requestParams.tests[testName].selected_features = Array.from(testParams.selected_features);
                }
            });

            $scope.loading.results = true;
            $http.post(getWebAppBackendUrl("stress-tests-config"), requestParams).then(function() {
                $http.get(getWebAppBackendUrl("compute")).then(function(response) {
                    angular.forEach(response.data, function(result) {
                        if (result.critical_samples) {
                            result.critical_samples.predList = result.critical_samples.predList.map(function(predList) {
                                predList = Object.entries(predList).map(function(entry) {
                                    const [testName, result] = entry;
                                    const displayName = $scope.uiState[testName].displayName;
                                    return `${displayName}: ${$filter("toFixedIfNeeded")(result, 3)}`;
                                });
                                predList.unshift($scope.modelInfo.predType === 'REGRESSION' ? "Predicted values" : "True class probas");
                                return predList;
                            });
                        }
                    });
                    $scope.results = response.data;
                    $scope.loading.results = false;
                }, function(e) {
                    $scope.loading.results = false;
                    $scope.createModal.error(e.data);
                });
            }, function(e) {
                $scope.loading.results = false;
                $scope.createModal.error(e.data);
            });
        }

        let features;
        $scope.loading.modelInfo = true;
        $http.get(getWebAppBackendUrl("model-info"))
            .then(function(response) {
                $scope.loading.modelInfo = false;
                $scope.modelInfo.targetClasses = response.data["target_classes"];
                $scope.modelInfo.predType = response.data["pred_type"];

                $scope.settings.perfMetric = response.data["metric"];
                $scope.METRIC_NAMES = MetricNames.availableMetrics($scope.modelInfo.predType);
                $scope.TEST_ORDER =  ["TARGET_SHIFT", "SUBPOPULATION_SHIFT", "FEATURE_PERTURBATION"];

                features = response.data["features"];

                // Only display relevant tests for the current model
                const featureNames = Object.keys(features);
                Object.keys($scope.settings.tests).forEach(function(testName) {
                    const constraints = CorruptionUtils.TEST_CONSTRAINTS[testName];
                    if (constraints.allowedFeatureTypes) {
                        $scope.uiState[testName].availableColumns = featureNames.filter(
                            function(name) {
                                return constraints.allowedFeatureTypes.includes(features[name]);
                            });
                        $scope.uiState[testName].available = $scope.uiState[testName].availableColumns.length;
                    }

                    if (constraints.allowedPredictionTypes) {
                        $scope.uiState[testName].available = constraints.allowedPredictionTypes.includes($scope.modelInfo.predType)
                    }
                });
        }, function(e) {
            $scope.loading.modelInfo = false;
            $scope.createModal.error(e.data);
        });
    })}
)();
