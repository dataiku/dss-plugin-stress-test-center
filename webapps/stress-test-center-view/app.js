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
        function metrics(metric, isRegression) {
            const shortName = MetricNames.shortName(metric);
            const longName = MetricNames.longName(metric);

            const perfVarDesc = ` is the difference in the model's ` +
                `${toLowerCaseIfNotAcronym(longName)} between the altered dataset and the ` +
                "unaltered one.";

            const resilienceDescClassif = " is the ratio of rows where the prediction is not " +
                "altered after the corruption.";

            const resilienceDescReg = " is the ratio of rows where the corruption does not " +
                "increase the prediction error.";

            const perfMetrics = [
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
                    name: "perf_var",
                    displayName: `${shortName} variation`,
                    longName: `${longName} variation`,
                    description: perfVarDesc
                },
                {
                    name: "corruption_resilience",
                    displayName: "Corruption resilience",
                    description: isRegression ? resilienceDescReg : resilienceDescClassif,
                    excludedStressTestTypes: ["TARGET_SHIFT"]
                }
            ];

            return function(testType) {
                return perfMetrics.filter(metric => !(metric.excludedStressTestTypes || []).includes(testType));
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
            longName: metric => metrics[metric].longName || metrics[metric].shortName,
            shortName: metric => metrics[metric].shortName || metric
        }
    });

    app.controller('VizController', function($scope, $http, ModalService, CorruptionUtils, MetricNames, $filter) {
        $scope.modal = {};
        $scope.removeModal = ModalService.remove($scope.modal);
        $scope.createModal = ModalService.create($scope.modal);

        $scope.CORRUPTION_TYPES = CorruptionUtils.types;
        $scope.TEST_NAMES = CorruptionUtils.TEST_NAMES;
        $scope.userFriendlyMetricName = MetricNames.longName

        $scope.loading = {};
        $scope.forms = {};
        $scope.tests = {
            perturbations: {
                Rebalance: {
                    needsTargetClasses: true,
                    params: { priors: {} }
                },
                MissingValues: {
                    allowedFeatureTypes: ["NUMERIC", "CATEGORY", "VECTOR", "TEXT"],
                    params: { samples_fraction: .5 },
                    selected_features: new Set()
                },
                Scaling: {
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
            if (!$scope.forms.GENERAL || $scope.forms.GENERAL.$invalid) return { canRun: false };
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
            $scope.perfMetrics = CorruptionUtils.metrics(
                requestParams.perfMetric,  $scope.modelInfo.predType === 'REGRESSION'
            );

            $scope.loading.results = true;
            $http.post(getWebAppBackendUrl("stress-tests-config"), requestParams).then(function() {
                $http.get(getWebAppBackendUrl("compute")).then(function(response) {
                    angular.forEach(response.data, function(result) {
                        if (result.critical_samples) {
                            result.critical_samples.predList = result.critical_samples.predList.map(function(predList) {
                                predList = Object.entries(predList).map(function(pred) {
                                    const [testName, result] = pred;
                                    return `${$scope.TEST_NAMES[testName]}: ${$filter("toFixedIfNeeded")(result, 3)}`;
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

                $scope.tests.perfMetric = response.data["metric"];
                $scope.METRIC_NAMES = MetricNames.availableMetrics($scope.modelInfo.predType);
                $scope.TEST_ORDER =  ["FEATURE_PERTURBATION"];
                if ($scope.modelInfo.predType !== 'REGRESSION') {
                    $scope.TEST_ORDER.unshift("TARGET_SHIFT")
                }

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
                        if ($scope.modelInfo.predType === 'REGRESSION') {
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
