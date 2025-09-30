(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        yellow - block
        blue - block
        orange - block
        red - block
        green - block
        purple - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable yellow)
        (clear yellow)
        (ontable blue)
        (clear blue)
        (ontable orange)
        (on red orange)
        (on green red)
        (clear green)
        (ontable purple)
        (clear purple)
        ; yellow
        ; blue
        ; orange -> red -> green
        ; purple
    )
    (:goal (and (on blue orange)
        (on purple green)
        (on yellow red)))
)
    