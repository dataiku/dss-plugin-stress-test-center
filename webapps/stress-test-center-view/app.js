let webAppConfig = dataiku.getWebAppConfig();
let modelId = webAppConfig['modelId'];
let versionId = webAppConfig['versionId'];

(function() {
    'use strict';
    app.controller('VizController', function($scope, $http, ModalService) {
        $scope.uiState = {
            selectedRow: "priorShift"
        };

        $scope.perturbations = {
            priorShift: {
                displayName: "Target distribution",
                params: {
                    affectedSamples: .5
                }
            },
            advAttack: {
                displayName: "Adversarial attack",
                params: {
                    affectedSamples: .5
                }
            },
            missingValues: {
                displayName: "Missing values enforcer",
                params: {
                    affectedSamples: .5
                }
            },
            scaling: {
                displayName: "Scaling perturbation",
                params: {
                    affectedSamples: .5
                }
            },
            replaceWord: {
                displayName: "Replace word",
                params: {
                    affectedSamples: .5
                }
            },
            typos: {
                displayName: "Typos",
                params: {
                    affectedSamples: .5
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

            const paramPs = $scope.perturbations.priorShift.$activated ? $scope.perturbations.priorShift.params.affectedSamples : 0;
            const paramAa = $scope.perturbations.advAttack.$activated ? $scope.perturbations.advAttack.params.affectedSamples : 0;
            const paramMv = $scope.perturbations.missingValues.$activated ? $scope.perturbations.missingValues.params.affectedSamples : 0;
            const paramS = $scope.perturbations.scaling.$activated ? $scope.perturbations.scaling.params.affectedSamples : 0;
            const paramT1 = $scope.perturbations.replaceWord.$activated ? $scope.perturbations.replaceWord.params.affectedSamples : 0;
            const paramT2 = $scope.perturbations.typos.$activated ? $scope.perturbations.typos.params.affectedSamples : 0;

            if (paramPs + paramAa + paramMv + paramS + paramT1 + paramT2 === 0) {
                $scope.uiState.loadingResult = false;
                return;
            };

            $http.get(getWebAppBackendUrl("compute/"+modelId+"/"+versionId+"?paramPS="+paramPs+"&paramAA="+paramAa+"&paramMV="+paramMv+"&paramS="+paramS+"&paramT1="+paramT1+"&paramT2="+paramT2))
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