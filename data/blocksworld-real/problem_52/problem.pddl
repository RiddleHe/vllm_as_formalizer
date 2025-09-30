(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        green - block
        purple - block
        blue - block
        orange - block
        red - block
        yellow - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable green)
        (on purple green)
        (clear purple)
        (ontable blue)
        (clear blue)
        (ontable orange)
        (on red orange)
        (on yellow red)
        (clear yellow)
        ; green -> purple
        ; blue
        ; orange -> red -> yellow
    )
    (:goal (and (on purple green)
        (on yellow red)
        (on orange blue)))
)
    