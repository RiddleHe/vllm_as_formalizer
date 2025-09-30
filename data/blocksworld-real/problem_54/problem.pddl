(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        yellow - block
        orange - block
        purple - block
        red - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (on yellow green)
        (clear yellow)
        (ontable orange)
        (clear orange)
        (ontable purple)
        (clear purple)
        (ontable red)
        (clear red)
        (ontable blue)
        (clear blue)
        ; green -> yellow
        ; orange
        ; purple
        ; red
        ; blue
    )
    (:goal (and (on blue red)
        (on green blue)
        (on purple green)
        (on orange purple)
        (on yellow orange)))
)
    