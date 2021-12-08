const webAppConfig = dataiku.getWebAppConfig();
const modelId = webAppConfig['modelId'];
const versionId = webAppConfig['versionId'];

(function() {
    'use strict';
    app.constant("CorruptionUtils", {
        metrics: {
            FEATURE_PERTURBATION: [
                {
                    name: "performance_variation",
                    displayName: "Performance variation"
                },
                {
                    name: "corruption_resilience",
                    displayName: "Corruption resilience"
                },
            ],
            TARGET_SHIFT: [
                {
                    name: "performance_variation",
                    displayName: "Performance variation"
                }
            ]
        },
        types: {
            FEATURE_PERTURBATION: {
                displayName: "Feature perturbations",
                description: "These stress tests corrupt the value of one or several features across randomly sampled rows."
            },
            TARGET_SHIFT: {
                displayName: "Target distribution shift",
                description: "This stress test resamples the dataset to match a desired distribution for the target column."
            }
        }
    });

    app.constant('MetricNames', {
        F1: "F1 score",
        ACCURACY: "Accuracy",
        PRECISION: "Precision",
        RECALL: "Recall",
        COST_MATRIX: "Cost matrix gain",
        ROC_AUC: "AUC",
        LOG_LOSS: "Log loss",
        CUMULATIVE_LIFT: "Cumulative lift",
        EVS: "Explained Variance Score",
        MAPE: "Mean Absolute Percentage Error",
        MAE: "Mean Absolute Error",
        MSE: "Mean Squared Error",
        RMSE: "Root Mean Square Error",
        RMSLE: "Root Mean Square Logarithmic Error",
        R2: "R2 score",
        CUSTOM: "A custom code metric"
    });

    app.controller('VizController', function($scope, $http, ModalService, CorruptionUtils, MetricNames) {
        $scope.modal = {};
        $scope.removeModal = ModalService.remove($scope.modal);
        $scope.createModal = ModalService.create($scope.modal);

        $scope.CORRUPTION_TYPES = CorruptionUtils.types;
        $scope.CORRUPTION_METRICS = CorruptionUtils.metrics;
        $scope.loading = {};
        $scope.forms = {};
        $scope.tests = {
            perturbations: {
                Rebalance: {
                    displayName: "Target distribution",
                    needsTargetClasses: true,
                    params: { priors: {} }
                },
                MissingValues: {
                    displayName: "Missing values",
                    allowedFeatureTypes: ["NUMERIC", "CATEGORY"],
                    params: { samples_fraction: .5 },
                    selected_features: new Set()
                },
                Scaling: {
                    displayName: "Scaling",
                    allowedFeatureTypes: ["NUMERIC"],
                    params: {
                        samples_fraction: .5,
                        scaling_factor: 10
                    },
                    selected_features: new Set()
                }
            },
            samples: 1,
            randomSeed: 65537
        };
        $scope.modelInfo = {};

        const featureTypesToIconClass = {
            NUMERIC: "numerical",
            CATEGORY: "icon-font"
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
                        $scope.loading.results = false;
                        $scope.results = response.data;
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

                $scope.modelInfo.metric = MetricNames[response.data["metric"].actual];
                if (response.data["metric"].initial === "CUSTOM") {
                    const warning_msg = "The corruption metrics computed by this webapp use the " +
                        "metric that was selected for model hyperparameter optimization. " +
                        `However custom metrics are not supported. ${$scope.modelInfo.metric} ` +
                        " will be leveraged instead.";

                    $scope.createModal.alert(warning_msg, "Warning");
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
                        if ($scope.modelInfo.isRegression) {
                            delete $scope.tests.perturbations[testName];
                        }
                    }
                });
        }, function(e) {
            $scope.createModal.error(e.data);
        });
    })}
)();
