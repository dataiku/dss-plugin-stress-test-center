let webAppConfig = dataiku.getWebAppConfig();
let modelId = webAppConfig['modelId'];
let versionId = webAppConfig['versionId'];

(function() {
    'use strict';
    app.controller('vizController', function($scope, $http, $timeout, ModalService) {
       $scope.activatePS = false;
       $scope.activateAA = false;
       $scope.activateMV = false;
       $scope.activateS = false;
       $scope.activateT1 = false;
       $scope.activateT2 = false

       $scope.highlightPS = true;
       $scope.highlightAA = false;
       $scope.highlightMV = false;
       $scope.highlightS = false;
       $scope.highlightT1 = false;
       $scope.highlightT2 = false;

       $scope.paramPS = 0.5;
       $scope.paramAA = 0.5;
       $scope.paramMV = 0.5;
       $scope.paramS = 0.5;
       $scope.paramT1 = 0.5;
       $scope.paramT2 = 0.5;

       var param_ps = 0;
       var param_aa = 0;
       var param_mv = 0;
       var param_s = 0;
       var paramT1 = 0;
       var paramT2 = 0;

        $scope.modal = {};
        $scope.removeModal = function(event) {
            if (ModalService.remove($scope.modal)(event)) {
                angular.element(".template").focus();
            }
        };
        $scope.createModal = ModalService.create($scope.modal);

        $scope.highlightDivPS = function(){
            $scope.highlightPS = true;
            $scope.highlightAA = false;
            $scope.highlightMV = false;
            $scope.highlightS = false;
            $scope.highlightT1 = false;
            $scope.highlightT2 = false;
        }

        $scope.highlightDivAA = function(){
            $scope.highlightPS = false;
            $scope.highlightAA = true;
            $scope.highlightMV = false;
            $scope.highlightS = false;
            $scope.highlightT1 = false;
            $scope.highlightT2 = false;
        }

        $scope.highlightDivMV = function(){
            $scope.highlightPS = false;
            $scope.highlightAA = false;
            $scope.highlightMV = true;
            $scope.highlightS = false;
            $scope.highlightT1 = false;
            $scope.highlightT2 = false;
        }

        $scope.highlightDivS = function(){
            $scope.highlightPS = false;
            $scope.highlightAA = false;
            $scope.highlightMV = false;
            $scope.highlightS = true;
            $scope.highlightT1 = false;
            $scope.highlightT2 = false;
        }

        $scope.highlightDivT1 = function(){
            $scope.highlightPS = false;
            $scope.highlightAA = false;
            $scope.highlightMV = false;
            $scope.highlightS = false;
            $scope.highlightT1 = true;
            $scope.highlightT2 = false;
        }

        $scope.highlightDivT2 = function(){
            $scope.highlightPS = false;
            $scope.highlightAA = false;
            $scope.highlightMV = false;
            $scope.highlightS = false;
            $scope.highlightT1 = false;
            $scope.highlightT2 = true;
        }

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
                $scope.highlightT1 = false;
                $scope.highlightT2 = false;
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
                $scope.highlightT1 = false;
                $scope.highlightT2 = false;
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
                $scope.highlightT1 = false;
                $scope.highlightT2 = false;
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
            $scope.highlightT1 = false;
            $scope.highlightT2 = false;
       };

       $scope.activateBoxT1 = function(){
            if ($scope.t1Active) {
                $scope.activateT1 = false;
            }
            else {
                console.log('Activate T1');
                $scope.activateT1 = true;
            }
            $scope.highlightS = false;
            $scope.highlightPS = false;
            $scope.highlightAA = false;
            $scope.highlightMV = false;
            $scope.highlightT1 = true;
            $scope.highlightT2 = false;
       };

        $scope.activateBoxT2 = function(){
            if ($scope.t2Active) {
                $scope.activateT2 = false;
            }
            else {
                console.log('Activate T2');
                $scope.activateT2 = true;
            }
            $scope.highlightS = false;
            $scope.highlightPS = false;
            $scope.highlightAA = false;
            $scope.highlightMV = false;
            $scope.highlightT1 = false;
            $scope.highlightT2 = true;
       };


       $scope.runAnalysis = function () {

             $scope.loadingResult = true;
             markRunning(true);

             if ($scope.activatePS) {
                param_ps = $scope.paramPS;
                console.log('Prior shift is chosen with param ', param_ps);
             }
             else {
                param_ps = 0;
             };

             if ($scope.activateAA) {
                param_aa = $scope.paramAA
                console.log('Adversarial attack is chosen with param ', param_aa);
             }
             else {
                param_aa = 0;
             };

             if ($scope.activateMV) {
                param_mv = $scope.paramMV
                console.log('Missing values is chosen with param ', param_mv);
             }
             else {
                param_mv = 0;
             }

            if ($scope.activateS) {
                param_s = $scope.paramS;
                console.log('Scaling is chosen with param ', param_s);
             }
             else {
                param_s = 0;
             }

            if ($scope.activateST1) {
                console.log('Text attack type 1 is chosen with param ', $scope.paramT1);
             };

            if ($scope.activateST2) {
                console.log('Text attack type 2 is chosen with param ', $scope.paramT2);
             };

             $('#error_message').html('');

            $http.get(getWebAppBackendUrl("compute/"+modelId+"/"+versionId+"?paramPS="+param_ps+"&paramAA="+param_aa+"&paramMV="+param_mv+"&paramS="+param_s))
                .then(function(response){
                    $scope.loadingResult = false;
                    console.log(response.data);
                    $scope.metrics = response.data['metrics'];
                    $scope.table_data = response.data['critical_samples'][0]
                    $scope.critical_samples = response.data['critical_samples']
                    $('.critical-sample-container').show();
                    $('.metric-container').show();
                    markRunning(false);
            }, function(e) {
                console.log(e);
                $scope.loadingResult = false;
                markRunning(false);
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