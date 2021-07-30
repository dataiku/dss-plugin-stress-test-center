let webAppConfig = dataiku.getWebAppConfig();
let modelId = webAppConfig['modelId'];
let versionId = webAppConfig['versionId'];

(function() {
    'use strict';
    app.controller('VizController', function($scope, $http, ModalService) {
        $scope.uiState = {
            selectedRow: "PRIOR_SHIFT"
        };

        $scope.perturbations = {
            PRIOR_SHIFT: {
                displayName: "Target distribution",
                params: {
                    samples_fraction: .5
                }
            },
            ADVERSARIAL: {
                displayName: "Adversarial attack",
                params: {
                    samples_fraction: .5
                }
            },
            MISSING_VALUES: {
                displayName: "Missing values enforcer",
                params: {
                    samples_fraction: .5
                }
            },
            SCALING: {
                displayName: "Scaling perturbation",
                params: {
                    samples_fraction: .5
                }
            },
            REPLACE_WORD: {
                displayName: "Replace word",
                params: {
                    samples_fraction: .5
                }
            },
            TYPOS: {
                displayName: "Typos",
                params: {
                    samples_fraction: .5
                }
            }
        }

        $scope.modal = {};
        $scope.removeModal = function(event) {
            if (ModalService.remove($scope.modal)(event)) {
                angular.element(".template").focus();
            }
        };
        $scope.createModal = ModalService.create($scope.modal);

       $scope.runAnalysis = function () {
            $scope.uiState.loadingResult = true;
            const perturbationsToCompute = {};
            for (let key in $scope.perturbations) {
                if ($scope.perturbations.$activated) {
                    perturbationsToCompute[key] = $scope.perturbations[key];
                }
            }

            $http.post(getWebAppBackendUrl("compute"), perturbationsToCompute)
                .then(function(response){
                    $scope.uiState.loadingResult = false;
                    $scope.metrics = response.data['metrics'];
                    $scope.critical_samples = response.data['critical_samples']
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


    })}
)();