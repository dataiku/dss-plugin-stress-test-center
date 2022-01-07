const webAppConfig = dataiku.getWebAppConfig();
const modelId = webAppConfig['modelId'];
const versionId = webAppConfig['versionId'];

(function() {
    'use strict';
    app.service("CorruptionUtils", function(MetricNames) {
        function metrics(metric, isRegression) {
            const [shortName, longName] = [
                MetricNames[metric.actual].shortName || metric.actual,
                MetricNames[metric.actual].longName
            ];

            let metricUsedDesc;
            if (metric.initial === "CUSTOM") {
                metricUsedDesc = "the default metric for "+
                `${isRegression ? "regression tasks (R2 score)" : "classification tasks (ROC AUC)"}.`;
            } else {
                metricUsedDesc = `the model's hyperparameter optimization metric (here, ${longName}).`;
            }

            const perfVarDesc = "Performance variation is the difference in the model's performance " +
            `between the altered and unaltered dataset. Performance is assessed via ${metricUsedDesc}`;

            const resilienceDescClassif = "Corruption resilience is the ratio of rows where " +
                "the prediction is not altered after the corruption.";

            const resilienceDescReg = "Corruption resilience is the ratio of rows where the " +
                "error between predicted and true values is not greater after the corruption.";

            return {
                FEATURE_PERTURBATION: [
                    {
                        name: "perf_before",
                        displayName: `${shortName} before`,
                        contextual: true
                    },
                    {
                        name: "perf_after",
                        displayName: `${shortName} after`,
                        contextual: true
                    },
                    {
                        name: "performance_variation",
                        displayName: `${shortName} variation`,
                        description: perfVarDesc
                    },
                    {
                        name: "corruption_resilience",
                        displayName: "Corruption resilience",
                        description: isRegression ? resilienceDescReg : resilienceDescClassif
                    },
                ],
                TARGET_SHIFT: [
                    {
                        name: "perf_before",
                        displayName: `${shortName} before`,
                        contextual: true
                    },
                    {
                        name: "perf_after",
                        displayName: `${shortName} after`,
                        contextual: true
                    },
                    {
                        name: "performance_variation",
                        displayName: `${shortName} variation`,
                        description: perfVarDesc
                    },
                ]
            };
        };

        return {
            metrics,
            types: {
                FEATURE_PERTURBATION: {
                    displayName: "Feature corruptions",
                    description: "Each of these independent tests corrupts one or several features across randomly sampled rows."
                },
                TARGET_SHIFT: {
                    displayName: "Target distribution shift",
                    description: "This stress test resamples the test set to match the desired distribution for the target column."
                }
            },
            TEST_NAMES: {
                _dku_stress_test_uncorrupted: "No corruption",
                Rebalance: "Shift target distribution",
                MissingValues: "Insert missing values",
                Scaling: "Multiply by a coefficient"
            }
        };
    });

    app.constant('MetricNames', {
        F1: {
            longName: "F1 score",
        },
        ACCURACY: {
            longName: "accuracy",
            shortName: "Accuracy"
        },
        PRECISION: {
            longName: "precision",
            shortName: "Precision"
        },
        RECALL: {
            longName: "recall",
            shortName: "Recall"
        },
        COST_MATRIX: {
            longName: "cost matrix gain",
            shortName: "Cost matrix gain"
        },
        ROC_AUC: {
            longName: "AUC",
            shortName: "AUC"
        },
        LOG_LOSS: {
            longName: "log loss",
            shortName: "Log loss"
        },
        CUMULATIVE_LIFT: {
            longName: "cumulative lift",
            shortName: "Lift"
        },
        EVS: {
            longName: "explained variance score",
        },
        MAPE: {
            longName: "mean absolute percentage error"
        },
        MAE: {
            longName: "mean absolute error"
        },
        MSE: {
            longName: "mean squared error"
        },
        RMSE: {
            longName: "root mean square error"
        },
        RMSLE: {
            longName: "root mean square logarithmic error"
        },
        R2: {
            longName: "R2 score"
        }
    });

    app.controller('VizController', function($scope, $http, ModalService, CorruptionUtils, $filter) {
        $scope.modal = {};
        $scope.removeModal = ModalService.remove($scope.modal);
        $scope.createModal = ModalService.create($scope.modal);

        $scope.CORRUPTION_TYPES = CorruptionUtils.types;

        $scope.loading = {};
        $scope.forms = {};
        $scope.tests = {
            perturbations: {
                Rebalance: {
                    displayName: "Shift target distribution", // TODO: remove (use CorruptionUtils.TEST_NAMES instead)
                    needsTargetClasses: true,
                    params: { priors: {} }
                },
                MissingValues: {
                    displayName: "Insert missing values",
                    allowedFeatureTypes: ["NUMERIC", "CATEGORY", "VECTOR", "TEXT"],
                    params: { samples_fraction: .5 },
                    selected_features: new Set()
                },
                Scaling: {
                    displayName: "Multiply by a coefficient",
                    allowedFeatureTypes: ["NUMERIC"],
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
        $scope.modelInfo = {};

        const featureTypesToIconClass = {
            NUMERIC: "numerical",
            CATEGORY: "icon-font",
            TEXT: "icon-italic",
            VECTOR: "vector"
        };

        $scope.featureToTypeIcon = function(feature) {
            return featureTypesToIconClass[features[feature]];
        }

        $scope.checkTestConfig = function() {
            if (!$scope.forms.SAMPLES || $scope.forms.SAMPLES.$invalid) return { canRun: false };
            const testEntries = Object.entries($scope.tests.perturbations);
            const validActivatedTests = testEntries.filter(function(entry) {
                const [testName, testSettings] = entry;
                return testSettings.$activated && $scope.forms[testName].$valid;
            });
            const invalidActivatedTests = testEntries.filter(function(entry) {
                const [testName, testSettings] = entry;
                return testSettings.$activated && $scope.forms[testName].$invalid;
            });
            return {
                canRun: validActivatedTests.length && !invalidActivatedTests.length,
                config: validActivatedTests
            }
        }

        $scope.runAnalysis = function () {
            const { canRun, config } = $scope.checkTestConfig();
            if (!canRun) return;
            const requestParams = Object.assign({}, $scope.tests);
            requestParams.perturbations = config.reduce(function(fullParams, currentEntry) {
                const [testName, testSettings] = currentEntry;
                fullParams[testName] = {
                    params: testSettings.params
                }
                if (testSettings.selected_features) { // test is a sample perturbation
                    fullParams[testName].selected_features = Array.from(testSettings.selected_features);
                }
                return fullParams;
            }, {});

            $scope.loading.results = true;
            $http.post(
                getWebAppBackendUrl("stress-tests-config"), requestParams).then(function() {
                $http.get(getWebAppBackendUrl("compute"))
                    .then(function(response) {
                        angular.forEach(response.data, function(result) {
                            if (result.critical_samples) {
                                result.critical_samples.predList = result.critical_samples.predList.map(function(predList) {
                                    predList = Object.entries(predList).map(function(pred) {
                                        const [testName, result] = pred;
                                        return `${CorruptionUtils.TEST_NAMES[testName]}: ${$filter("toFixedIfNeeded")(result, 3)}`;
                                    });
                                    predList.unshift($scope.modelInfo.isRegression ? "Predicted values" : "True class probas");
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
                $scope.modelInfo.isRegression = !$scope.modelInfo.targetClasses.length;

                $scope.CORRUPTION_METRICS = CorruptionUtils.metrics(response.data["metric"],  $scope.modelInfo.isRegression);

                features = response.data["features"];

                // Only display relevant tests for the current model
                const featureNames = Object.keys(features);
                Object.keys($scope.tests.perturbations).forEach(function(testName) {
                    const testConfig = $scope.tests.perturbations[testName];
                    if (testConfig.allowedFeatureTypes) {
                        testConfig.availableColumns = featureNames.filter(function(name) {
                            return testConfig.allowedFeatureTypes.includes(features[name]);
                        });
                        if (!testConfig.availableColumns.length) {
                            delete $scope.tests.perturbations[testName];
                        }
                    }
                    if (testConfig.needsTargetClasses) {
                        if ($scope.modelInfo.isRegression) {
                            delete $scope.tests.perturbations[testName];
                        }
                    }
                });
        }, function(e) {
            $scope.loading.modelInfo = false;
            $scope.createModal.error(e.data);
        });
    })}
)();
