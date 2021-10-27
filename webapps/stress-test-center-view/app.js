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

        $scope.runAnalysis = function () {
            const perturbationsToCompute = {};
            for (let key in $scope.perturbations) {
                const perturbation = $scope.perturbations[key];
                if (perturbation.$activated) {
                    if (perturbation.testType === "SAMPLE_PERTURBATION") {
                        if (!perturbation.selected_features.size) continue;
                        perturbationsToCompute[key] = {params: perturbation.params};
                        perturbationsToCompute[key].selected_features = Array.from(perturbation.selected_features);
                    } else {
                        if (!$scope.perturbations[key].params.cl) continue; // TODO: cleaner
                        perturbationsToCompute[key] = {params: perturbation.params};
                    }
                }
            }
            if (!Object.keys(perturbationsToCompute).length) return;

            $scope.uiState.loadingResult = true;
            $http.post(getWebAppBackendUrl("stress-tests-config"), perturbationsToCompute)
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
