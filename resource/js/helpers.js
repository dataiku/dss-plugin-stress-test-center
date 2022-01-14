'use strict';

app.service("ModalService", function() {
    const remove = function(config) {
        return function(event) {
            if (event && !event.target.className.includes("modal-background")) return false;
            for (const key in config) {
                delete config[key];
            }
            return true;
        }
    };
    return {
        create: function(config) {
            return {
                confirm: function(msg, title, confirmAction) {
                    Object.assign(config, {
                        type: "confirm",
                        msg: msg,
                        title: title,
                        confirmAction: confirmAction
                    });
                },
                error: function(msg) {
                    Object.assign(config, {
                        type: "error",
                        msg: msg,
                        title: "Backend error"
                    });
                },
                alert: function(msg, title) {
                    Object.assign(config, {
                        type: "alert",
                        msg: msg,
                        title: title
                    });
                },
                prompt: function(inputLabel, confirmAction, res, title, msg, attrs) {
                    Object.assign(config, {
                        type: "prompt",
                        inputLabel: inputLabel,
                        promptResult: res,
                        title: title,
                        msg: msg,
                        conditions: attrs,
                        confirmAction: function() {
                            confirmAction(config.promptResult);
                        }
                    });
                }
            };
        },
        remove: remove
    }
});

app.directive("modalBackground", function($compile) {
    return {
        scope: true,
        restrict: "C",
        templateUrl: "/plugins/model-stress-test/resource/templates/modal.html",
        link: function(scope, element) {
            if (scope.modal.conditions) {
                const inputField = element.find("input");
                for (const attr in scope.modal.conditions) {
                    inputField.attr(attr, scope.modal.conditions[attr]);
                }
                $compile(inputField)(scope);
            }
        }
    }
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

            scope.display = scope.display || (item => item);

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
                if (!scope.item) return "Select a " + scope.itemName;
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
                if (!scope.keyOptions) return;
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

            scope.$watch("map", function(nv, ov) {
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
        template: `<i class="icon-info-sign icon--hoverable" ng-mouseover="showTooltip()" ng-mouseleave="showTooltip()">
            <div class="help-text__tooltip tooltip--hidden">
                <i class="icon-info-sign"></i>
                {{ helpText }}
            </div>
        </i>`,
        link: function(scope, elem) {
            scope.showTooltip = function() {
                const top = elem[0].getBoundingClientRect().top;
                const tooltip = elem.find(".help-text__tooltip");
                tooltip.css("top", (top - 8) + "px");
                tooltip.toggleClass("tooltip--hidden");
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
