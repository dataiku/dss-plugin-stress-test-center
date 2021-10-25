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

        $scope.uiState = {
            openFeatureSelectors: {}
        };
        $scope.perturbations = {};
        $scope.modelInfo = {};

        $scope.openFeatureSelector = function(perturbation, event) {
            $scope.uiState.openFeatureSelectors[perturbation] = !$scope.uiState.openFeatureSelectors[perturbation];
            event.stopPropagation();
        }

        $scope.getFeatureDropdownPlaceholder = function(perturbation) {
            const nrSelectedItems = $scope.perturbations[perturbation].available_columns.filter(col => col.$selected).length;
            if (!nrSelectedItems) return "Select features";
            return nrSelectedItems + " feature" + (nrSelectedItems > 1 ? "s" : "");
        }

        $scope.runAnalysis = function () {
            $scope.uiState.loadingResult = true;
            const perturbationsToCompute = {};
            for (let key in $scope.perturbations) {
                if ($scope.perturbations[key].$activated) {
                    perturbationsToCompute[key] = Object.assign({}, $scope.perturbations[key]);
                    perturbationsToCompute[key].available_columns = (perturbationsToCompute[key].available_columns || []).filter(col => col.$selected).map(col => col.name);
                }
            }
            $http.post(getWebAppBackendUrl("stress-tests-config"), perturbationsToCompute)
                .then(function() {
                    $http.get(getWebAppBackendUrl("compute"))
                        .then(function(response) {
                            $scope.uiState.loadingResult = false;
                            $scope.metrics = response.data['metrics'];
                            $scope.critical_samples = response.data['critical_samples']
                            $scope.uncertainties = response.data['uncertainties']
                    }, function(e) {
                        $scope.uiState.loadingResult = false;
                        $scope.createModal.error(e.data);
                    });
                }, function(e) {
                    $scope.uiState.loadingResult = false;
                    $scope.createModal.error(e.data);
                });
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

                const columns = response.data["columns"];
                $scope.perturbations.MISSING_VALUES = {
                    displayName: "Missing values enforcer",
                    params: {
                        samples_fraction: .5
                    },
                    available_columns: columns.filter(col => ["CATEGORY", "NUMERIC"].includes(col.feature_type))
                };

                $scope.perturbations.SCALING = {
                    displayName: "Scaling perturbation",
                    params: {
                        samples_fraction: .5
                    },
                    available_columns: columns.filter(col => ["NUMERIC"].includes(col.feature_type))
                };
        }, function(e) {
            $scope.createModal.error(e.data);
        });
    })}
)();
