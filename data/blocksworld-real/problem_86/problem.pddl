(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        purple - block
        orange - block
        yellow - block
        blue - block
        green - block
        red - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable purple)
        (clear purple)
        (ontable orange)
        (on yellow orange)
        (clear yellow)
        (ontable blue)
        (clear blue)
        (ontable green)
        (on red green)
        (clear red)
        ; purple
        ; orange -> yellow
        ; blue
        ; green -> red
    )
    (:goal (and (on purple blue)
        (on orange purple)
        (on green orange)
        (on yellow red)))
)
    