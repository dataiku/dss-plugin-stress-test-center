const webAppConfig = dataiku.getWebAppConfig();
const modelId = webAppConfig['modelId'];
const versionId = webAppConfig['versionId'];

(function() {
    'use strict';
    app.controller('VizController', function($scope, $http, ModalService) {
        $scope.modal = {};
        $scope.removeModal = function(event) {
            if (ModalService.remove($scope.modal)(event)) {
                angular.element(".template").focus();
            }
        };
        $scope.createModal = ModalService.create($scope.modal);

        $scope.loading = {};
        $scope.forms = {};
        $scope.tests = {
            perturbations: {
                MISSING_VALUES: {
                    displayName: "Missing values",
                    testType: "SAMPLE_PERTURBATION",
                    params: { samples_fraction: .5 },
                    selected_features: new Set()
                },
                SCALING: {
                    displayName: "Scaling",
                    testType: "SAMPLE_PERTURBATION",
                    params: { samples_fraction: .5 },
                    selected_features: new Set()
                }
            },
            samples: 1
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
            if ($scope.forms.SAMPLES.$invalid) return { canRun: false };
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
            const perturbationsConfig = config.reduce(function(fullParams, currentEntry) {
                const [testName, testSettings] = currentEntry;
                fullParams[testName] = {
                    params: testSettings.params
                }
                if (testSettings.testType === "SAMPLE_PERTURBATION") {
                    fullParams[testName].selected_features = Array.from(testSettings.selected_features);
                }
                return fullParams;
            }, {});

            $scope.loading.results = true;
            $http.post(
                getWebAppBackendUrl("stress-tests-config"),
                { perturbations: perturbationsConfig, samples: $scope.tests.samples}
            ).then(function() {
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
                if ($scope.modelInfo.targetClasses.length) {
                    $scope.tests.perturbations.PRIOR_SHIFT = {
                        displayName: "Target distribution",
                        testType: "SUBPOPULATION_PERTURBATION",
                        params: { samples_fraction: .5 }
                    };
                }

                features = response.data["features"];
                const featureNames = Object.keys(features);
                $scope.tests.perturbations.MISSING_VALUES.availableColumns = featureNames.filter(function(name) {
                    return ["NUMERIC", "CATEGORY"].includes(features[name]);
                });

                $scope.tests.perturbations.SCALING.availableColumns = featureNames.filter(function(name) {
                    return ["NUMERIC"].includes(features[name]);
                });
        }, function(e) {
            $scope.createModal.error(e.data);
        });
    })}
)();
