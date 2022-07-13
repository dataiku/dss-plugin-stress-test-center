'use strict';

app.service("ModalService", function($compile, $http) {
    const DEFAULT_MODAL_TEMPLATE = "/plugins/model-stress-test/resource/templates/modal.html";

    function create(scope, config, templateUrl=DEFAULT_MODAL_TEMPLATE) {
        $http.get(templateUrl).then(function(response) {
            const template = response.data;
            const newScope = scope.$new();
            const element = $compile(template)(newScope);

            angular.extend(newScope, config);

            newScope.close = function(event) {
                if (event && !event.target.className.includes("modal-background")) return;
                element.remove();
                newScope.$emit("closeModal");
            };

            if (newScope.promptConfig && newScope.promptConfig.conditions) {
                const inputField = element.find("input");
                for (const attr in newScope.promptConfig.conditions) {
                    inputField.attr(attr, newScope.promptConfig.conditions[attr]);
                }
                $compile(inputField)(newScope);
            }

            angular.element("body").append(element);
        });
    };
    return {
        createBackendErrorModal: function(scope, errorMsg) {
            create(scope, {
                title: 'Backend error',
                msgConfig: { error: true, msg: errorMsg }
            }, DEFAULT_MODAL_TEMPLATE);
        },
        create
    };
});

app.directive("spinner", function () {
    return {
        template: "<div class='spinner-container'></div>",
        link: function (scope, element) {
            var opts = {
                lines: 6,
                length: 0,
                width: 10,
                radius: 10,
                corners: 1,
                rotate: 0,
                color: '#fff',
                speed: 2,
                trail: 60,
                shadow: false,
                hwaccel: false,
                className: 'spinner',
                zIndex: 2e9,
                top: '10px',
                left: '10px'
             };
             const spinner = new Spinner(opts);
             spinner.spin(element[0].childNodes[0]);
        }
    }
});

app.directive("customDropdown", function() {
    return {
        scope: {
            form: '=?',
            itemImage: '=?',
            label: '@',
            itemName: '@',
            item: '=',
            items: '=',
            possibleValues: '=',
            notAvailableValues: '=',
            onChange: '=',
            display: '=?'
        },
        restrict: 'A',
        templateUrl:'/plugins/model-stress-test/resource/templates/custom-dropdown.html',
        link: function(scope, elem, attrs) {
            const VALIDITY = "dropdown-not-empty" + (attrs.id ? ("__" + attrs.id) : "");
            function setValidity() {
                if (!scope.form) return;
                scope.form.$setValidity(VALIDITY, !!scope.item || !!(scope.items || {}).size);
            }
            setValidity();

            scope.display = scope.display || (item => item === "__dku_missing_value__" ? "" : item);

            scope.canBeSelected = function(item) {
                if (!scope.notAvailableValues) return true;
                return item === scope.item || !(item in scope.notAvailableValues);
            };

            const isMulti = !!attrs.items;
            scope.isSelected = function(value) {
                if (isMulti) {
                    return scope.items.has(value);
                }
                return scope.item === value;
            };

            scope.updateSelection = function(value, event) {
                if (isMulti) {
                    if (scope.isSelected(value)) {
                        scope.items.delete(value);
                    } else {
                        scope.items.add(value);
                    }
                    event.stopPropagation();
                } else {
                    if (scope.item === value) return;
                    if (scope.onChange) {
                        scope.onChange(value, scope.item, elem);
                    }
                    scope.item = value;
                }
                setValidity();
            };

            scope.getPlaceholder = function() {
                if (isMulti) {
                    if (!(scope.items || {}).size) return "Select " + scope.itemName + "s";
                    return scope.items.size + " " + scope.itemName + (scope.items.size > 1 ? "s" : "");
                }
                if (scope.item == null) return "Select a " + scope.itemName;
                return scope.display(scope.item);
            };

            scope.toggleDropdown = function() {
                scope.isOpen = !scope.isOpen;
            };

            const dropdownElem = elem.find(".custom-dropdown");
            const labelElem = elem.find(".label-text");
            scope.$on("closeDropdowns", function(e, target) {
                if ((target) && ( angular.element(target).closest(dropdownElem)[0]
                    || angular.element(target).closest(labelElem)[0] )) { return; }
                scope.isOpen = false;
            });

            scope.$on("$destroy", function() {
                scope.form && scope.form.$setValidity(VALIDITY, true);
            });
        }
    }
})

// For now, key can only be one of a preset of values (dropdown)
// & the value field only accepts numbers
app.directive("keyValueList", function($timeout) {
    return {
        scope: {
            keyOptions: '=',
            map: '=',
            keyLabel: '@',
            keyItemLabel: '@',
            valueLabel: '@',
            valueRange: '@',
            step: '=',
            defaultValue: '='
        },
        restrict: 'A',
        templateUrl:'/plugins/model-stress-test/resource/templates/key-value-list.html',
        link: function(scope) {
            scope.step = scope.step || "any";
            [ scope.valueMin, scope.valueMax ] = scope.$eval(scope.valueRange) || [null, null];

            scope.deleteListItem = function(index) {
                const removedKey = scope.keys.splice(index, 1)[0];
                delete scope.map[removedKey];
            };

            scope.canAddListItem = function() {
                if (!scope.keyOptions || !scope.keys) return;
                return scope.keys.length < scope.keyOptions.length - 1;
            };

            scope.addListItem = function() {
                scope.keys.push(null);
            };

            scope.dropdownChange = function(newValue, oldValue, keyElem) {
                scope.map[newValue] = scope.map[oldValue] || scope.defaultValue;
                delete scope.map[oldValue];
                $timeout(function() {
                    const valueElem = keyElem.parent().find(".key-value-element__value")[0];
                    valueElem.focus();
                });
            };

            scope.$watch("map", function() {
                const keys = Object.keys(scope.map);
                if (keys.length) {
                    scope.keys = keys;
                } else {
                    scope.keys = [null];
                }
            });
        }
    }
});

app.directive("helpIcon", function () {
    return {
        restrict: 'E',
        scope: {
            helpText: '@',

        },
        template: `<i class="icon-info-sign icon--hoverable" ng-mouseover="toggleTooltip(true)" ng-mouseleave="toggleTooltip(false)">
            <div class="help-text__tooltip tooltip--hidden">
                <i class="icon-info-sign"></i>
                {{ helpText }}
            </div>
        </i>`,
        link: function(scope, elem) {
            scope.toggleTooltip = function(show) {
                const top = elem[0].getBoundingClientRect().top;
                const tooltip = elem.find(".help-text__tooltip");
                tooltip.css("top", (top - 8) + "px");
                tooltip.toggleClass("tooltip--hidden", !show);
            };
        }
    }
});

app.filter("toFixedIfNeeded", function() {
    return function(number, decimals) {
        const lowerBound = 5 * Math.pow(10, -(decimals + 1));
        if (number && Math.abs(number) < lowerBound) {
            return "< " + lowerBound*Math.sign(number);
        }

        if(Math.round(number) !== number) {
            return parseFloat(number.toFixed(decimals));
        }
        return number;
    }
});

app.directive('focusHere', function ($timeout) {
    return {
        restrict: 'A',
        link: function (scope, element) {
            $timeout(function() {
                element[0].focus();
            });
        }
    };
});
