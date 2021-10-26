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
            openCustomDropdowns: {}
        };
        $scope.perturbations = {};
        $scope.modelInfo = {};

        $scope.closeAllDropdowns = function() {
            if (Object.values($scope.uiState.openCustomDropdowns)
                .reduce((openDropdown, atLeastOneOpenDropdown) => openDropdown || atLeastOneOpenDropdown, false)) {
                $scope.uiState.openCustomDropdowns = {};
            }
        }

        $scope.openCustomDropdown = function(perturbation, event) {
            const isOpen = $scope.uiState.openCustomDropdowns[perturbation];
            if (!isOpen) {
                $scope.closeAllDropdowns()
            }
            $scope.uiState.openCustomDropdowns[perturbation] = !isOpen;
            event.stopPropagation();
        }

        $scope.getFeatureDropdownPlaceholder = function(perturbation) {
            if (!$scope.perturbations[perturbation]) return;
            const nrSelectedItems = $scope.perturbations[perturbation].available_columns.filter(col => col.$selected).length;
            if (!nrSelectedItems) return "Select features";
            return nrSelectedItems + " feature" + (nrSelectedItems > 1 ? "s" : "");
        }

        $scope.showTooltip = function($event) {
            const top = $event.target.getBoundingClientRect().top;
            const tooltip = angular.element($event.target).find(".settings__help-text");
            tooltip.css("top", (top - 8) + "px");
            tooltip.toggleClass("tooltip--hidden");
        }

        $scope.runAnalysis = function () {
            const perturbationsToCompute = {};
            for (let key in $scope.perturbations) {
                if ($scope.perturbations[key].$activated) {
                    if (key === 'PRIOR_SHIFT') { // TODO: cleaner check
                        if ($scope.perturbations.PRIOR_SHIFT.params.cl) {
                            perturbationsToCompute[key] = Object.assign({}, $scope.perturbations[key]);
                        }
                    } else {
                        const samplePerturbation = Object.assign({}, $scope.perturbations[key]);
                        samplePerturbation.available_columns = samplePerturbation.available_columns.filter(col => col.$selected).map(col => col.name);
                        if (samplePerturbation.available_columns.length) {
                            perturbationsToCompute[key] = samplePerturbation;
                        }
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
