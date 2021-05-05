let webAppConfig = dataiku.getWebAppConfig();
let modelId = webAppConfig['modelId'];
let versionId = webAppConfig['versionId'];

(function() {
    'use strict';
    app.controller('vizController', function($scope, $http, $timeout) {
       $scope.activatePS = false;
       $scope.activateAA = false;
       $scope.activateMV = false;
       $scope.activateS = false;

       $scope.highlightPS = true;
       $scope.highlightAA = false;
       $scope.highlightMV = false;
       $scope.highlightS = false;

       var choice_list = {
        'highlight_ps': $scope.highlightPS,
        'highlight_aa': $scope.highlightAA,
        'highlight_mv': $scope.highlightMV,
        'highlight_s': $scope.highlightS
        };

       $scope.activateBoxPS = function(){
            if ($scope.priorShiftActive) {
                $scope.activatePS = false;
            }
            else {
                console.log('Activate PS');
                $scope.activatePS = true;
            }
                $scope.highlightPS = true;
                $scope.highlightAA = false;
                $scope.highlightMV = false;
                $scope.highlightS = false;
       };

       $scope.activateBoxAA = function(){
            if ($scope.aaActive) {
                $scope.activateAA = false;
            }
            else {
                console.log('Activate AA');
                $scope.activateAA = true;
            }
                $scope.highlightAA = true;
                $scope.highlightPS = false;
                $scope.highlightMV = false;
                $scope.highlightS = false;
       };

        $scope.activateBoxMV = function(){
            if ($scope.mvActive) {
                $scope.activateMV = false;
            }
            else {
                console.log('Activate MV');
                $scope.activateMV = true;
            }
                $scope.highlightMV = true;
                $scope.highlightPS = false;
                $scope.highlightAA = false;
                $scope.highlightS = false;
       };

        $scope.activateBoxS = function(){
            if ($scope.sActive) {
                $scope.activateS = false;
            }
            else {
                console.log('Activate S');
                $scope.activateS = true;
            }
            $scope.highlightS = true;
            $scope.highlightPS = false;
            $scope.highlightAA = false;
            $scope.highlightMV = false;
            $scope.highlightS = true;
       };

       $scope.runAnalysis = function () {
             markRunning(true);
             $('#error_message').html('');
            $http.get(getWebAppBackendUrl("compute/"+modelId+"/"+versionId))
                .then(function(response){
                    console.log(response.data);
                    $scope.metrics = {}
                    $scope.metrics['prior_shift'] = response.data['metrics']['PRIOR_SHIFT'];
                    $scope.metrics['adversarial'] = response.data['metrics']['ADVERSARIAL'];
                    $scope.metrics['missing_values'] = response.data['metrics']['MISSING_VALUES'];
                    $scope.metrics['scaling'] = response.data['metrics']['SCALING'];

                    $scope.table_data = response.data['critical_samples'][0]
                    $scope.critical_samples = response.data['critical_samples']
                    markRunning(false);
            }, function(e) {
                console.log(e);
                markRunning(false);
                $scope.createModal.error(e.data);
            });
        }


    })}
)();