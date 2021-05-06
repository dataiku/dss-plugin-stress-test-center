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
       $scope.activateT1 = false;
       $scope.activateT2 = false

       $scope.highlightPS = true;
       $scope.highlightAA = false;
       $scope.highlightMV = false;
       $scope.highlightS = false;
       $scope.highlightT1 = false;
       $scope.highlightT2 = false;

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
             markRunning(true);

             if ($scope.activatePS) {
                console.log('Prior shift is chosen with param ', $scope.paramPS);
             };

             if ($scope.activateAA) {
                console.log('Adversarial attack is chosen with param ', $scope.paramAA);
             };

             if ($scope.activateMV) {
                console.log('Missing values is chosen with param ', $scope.paramMV);
             };

            if ($scope.activateS) {
                console.log('Scaling is chosen with param ', $scope.paramS);
             };

            if ($scope.activateST1) {
                console.log('Text attack type 1 is chosen with param ', $scope.paramT1);
             };

            if ($scope.activateST2) {
                console.log('Text attack type 2 is chosen with param ', $scope.paramT2);
             };

             $('#error_message').html('');

            var paramPS = 0.2;
            var paramAA = 0.1;
            var paramMV = 0.45;
            var paramS = 0.55;
            var paramT1 = 0.8;
            var paramT2 = 0.25;


            $http.get(getWebAppBackendUrl("compute/"+modelId+"/"+versionId+"?paramPS="+paramPS+"&paramAA="+paramAA+"&paramMV="+paramMV+"&paramS="+paramS))
                .then(function(response){
                    console.log(response.data);
                    $scope.metrics = response.data['metrics'];
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