let webAppConfig = dataiku.getWebAppConfig();
let modelId = webAppConfig['modelId'];
let versionId = webAppConfig['versionId'];

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

        $scope.uiState = {};
        $scope.perturbations = {};
        $scope.modelInfo = {};
        $scope.samples = 100;

        const featureTypesToIconClass = {
            NUMERIC: "numerical",
            CATEGORY: "icon-font"
        };

        $scope.featureToTypeIcon = function(feature) {
            return featureTypesToIconClass[features[feature]];
        }

        $scope.checkTestConfig = function() {
            if ($scope.settings_SAMPLES.$invalid) return { canRun: false };
            const testEntries = Object.entries($scope.perturbations);
            const validActivatedTests = testEntries.filter(function(entry) {
                const [testName, testSettings] = entry;
                return testSettings.$activated && $scope["settings_" + testName].$valid;
            });
            const invalidActivatedTests = testEntries.filter(function(entry) {
                const [testName, testSettings] = entry;
                return testSettings.$activated && $scope["settings_" + testName].$invalid;
            });
            return {
                canRun: validActivatedTests.length && !invalidActivatedTests.length,
                config: validActivatedTests
            }
        }

        $scope.runAnalysis = function () {
            const { canRun, config} = $scope.checkTestConfig();
            if (!canRun) return;
            const requestParams = config.reduce(function(fullParams, currentEntry) {
                const [testName, testSettings] = currentEntry;
                fullParams[testName] = {
                    params: testSettings.params
                }
                if (testSettings.testType === "SAMPLE_PERTURBATION") {
                    fullParams[testName].selected_features = Array.from(testSettings.selected_features);
                }
                return fullParams;
            }, {});

            $scope.uiState.loadingResult = true;
            $http.post(getWebAppBackendUrl("stress-tests-config"), requestParams)
                .then(function() {
                    $http.get(getWebAppBackendUrl("compute"))
                        .then(function(response) {
                            $scope.uiState.loadingResult = false;
                            $scope.results = response.data;
                    }, function(e) {
                        $scope.uiState.loadingResult = false;
                        $scope.createModal.error(e.data);
                    });
                }, function(e) {
                    $scope.uiState.loadingResult = false;
                    $scope.createModal.error(e.data);
                });
        }

        let features;
        $http.get(getWebAppBackendUrl("model-info"))
            .then(function(response){
                $scope.modelInfo.targetClasses = response.data["target_classes"];
                if ($scope.modelInfo.targetClasses.length) {
                    $scope.perturbations.PRIOR_SHIFT = {
                        displayName: "Target distribution",
                        testType: "SUBPOPULATION_PERTURBATION",
                        params: {
                            samples_fraction: .5
                        }
                    };
                }

                features = response.data["features"];
                const featureNames = Object.keys(features);
                $scope.perturbations.MISSING_VALUES = {
                    displayName: "Missing values",
                    availableColumns: featureNames.filter(name => ["NUMERIC", "CATEGORY"].includes(features[name])),
                    testType: "SAMPLE_PERTURBATION",
                    params: {
                        samples_fraction: .5
                    },
                    selected_features: new Set()
                };

                $scope.perturbations.SCALING = {
                    displayName: "Scaling",
                    availableColumns: featureNames.filter(name => ["NUMERIC"].includes(features[name])),
                    testType: "SAMPLE_PERTURBATION",
                    params: {
                        samples_fraction: .5
                    },
                    selected_features: new Set()
                };
        }, function(e) {
            $scope.createModal.error(e.data);
        });
    })}
)();
