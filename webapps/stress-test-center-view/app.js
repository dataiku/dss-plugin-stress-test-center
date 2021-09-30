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

        $scope.runAnalysis = function () {
            $scope.uiState.loadingResult = true;
            const perturbationsToCompute = {};
            for (let key in $scope.perturbations) {
                if ($scope.perturbations[key].$activated) {
                    perturbationsToCompute[key] = $scope.perturbations[key];
                }
            }
            $http.post(getWebAppBackendUrl("stress-tests-config"), perturbationsToCompute)
                .then(function(response) {
                    $http.get(getWebAppBackendUrl("compute"))
                        .then(function(response){
                            $scope.uiState.loadingResult = false;
                            $scope.metrics = response.data['metrics'];
                            $scope.critical_samples = response.data['critical_samples']
                    }, function(e) {
                        $scope.uiState.loadingResult = false;
                        $scope.createModal.error(e.data);
                    });
                }, function(e) {
                    $scope.uiState.loadingResult = false;
                    $scope.createModal.error(e.data);
                });

            $scope.filterUncertainty = function(item) {
                var result = {};
                for (let k in item) {
                    if (k != 'uncertainty') {
                        result[k] = item[k];
                    }
                }
                return result;
            }
        }

        $http.get(getWebAppBackendUrl("model-info"))
            .then(function(response){
                $scope.modelInfo.targetClasses = response.data["target_classes"];
                if ($scope.modelInfo.targetClasses.length) {
                    $scope.modelInfo.isClassification = true;
                    $scope.uiState.selectedRow = "PRIOR_SHIFT";
                    $scope.perturbations.PRIOR_SHIFT = {
                        displayName: "Target distribution perturbation",
                        params: {
                            samples_fraction: .5
                        }
                    };
                } else {
                    $scope.uiState.selectedRow = "MISSING_VALUES";
                }

                $scope.perturbations.MISSING_VALUES = {
                    displayName: "Missing values enforcer",
                    params: {
                        samples_fraction: .5
                    }
                };
                $scope.perturbations.SCALING = {
                    displayName: "Scaling perturbation",
                    params: {
                        samples_fraction: .5
                    }
                };
        }, function(e) {
            $scope.createModal.error(e.data);
        });
    })}
)();
